"""Measure same-predicted-class ray coverage across training checkpoints.

Point 0 selects the saved initial state. Point k selects the final batch
checkpoint from zero-based epoch k-1. Evaluation images and the evaluation
training split use seed 42 across runs to hold the measurement pool fixed.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

sys.path.insert(0, ".")

from experiments.gi_coverage.test_to_train.mnist_test_to_all_classes_gi_coverage import (
    load_model_from_config,
    cluster_train_by_labels,
    get_image_by_index,
)
from src.analysis.gi_calculation import compute_GIs_for_sample

SUMMARY_COLUMNS = [
    "arch", "run_type", "run_id", "point", "n_samples",
    "mean_gi", "mean_rc", "median_rc", "std_rc",
]


def resolve_checkpoint_for_point(checkpoint_dir, point):
    """point 0 -> the pre-training checkpoint; point k>=1 -> end of training epoch k-1."""
    if point == 0:
        path = os.path.join(checkpoint_dir, "model_epoch0_batch0.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"expected initial checkpoint at {path}")
        return path

    epoch_num = point - 1
    pattern = os.path.join(checkpoint_dir, f"model_epoch{epoch_num}_batch*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint found for point={point} (epoch_num={epoch_num}) matching {pattern}"
        )

    def batch_idx(path):
        m = re.search(r"_batch(\d+)\.pth$", path)
        return int(m.group(1))

    return max(candidates, key=batch_idx)


def load_fixed_eval_split(data_path, val_split):
    """Fresh MNIST train/val split + transform, seeded independently of training."""
    np.random.seed(42)
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = MNIST(root=data_path, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_path, train=False, transform=transform, download=True)
    val_size = int(val_split * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset


def sample_test_indices(test_dataset, k, total_classes=10):
    np.random.seed(42)
    test_indices = []
    for c in range(total_classes):
        class_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label == c]
        test_indices.extend(np.random.choice(class_indices, size=k, replace=False))
    return test_indices


def evaluate_checkpoint(model, train_dataset, test_dataset, test_indices, train_indices_by_class, device,
                         step_size, max_eps):
    rows = []
    for idx in test_indices:
        test_img, true_label = test_dataset[idx]
        test_img_dev = test_img.to(device).unsqueeze(0)
        pred = torch.argmax(model(test_img_dev)).item()

        train_indices = train_indices_by_class.get(pred, [])
        if not train_indices:
            rows.append({"sample_id": idx, "true_label": true_label, "predicted_class": pred,
                         "coverage": 0.0, "mean_gi": 0.0})
            continue

        train_imgs = torch.stack([get_image_by_index(train_dataset, i) for i in train_indices]).to(device)
        gis, _ = compute_GIs_for_sample(
            model=model, sample=test_img.squeeze(0), normalized=False,
            class_samples=train_imgs, correct_class=pred,
            step_size=step_size, max_eps=max_eps, device=device,
        )
        coverage = 100.0 * float(np.sum(gis > 1.0)) / len(gis) if gis.size > 0 else 0.0
        mean_gi = float(np.mean(gis)) if gis.size > 0 else 0.0
        rows.append({"sample_id": idx, "true_label": true_label, "predicted_class": pred,
                     "coverage": coverage, "mean_gi": mean_gi})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--points", required=True, help="comma-separated epochs-trained counts, e.g. 0,5,10,15,20")
    parser.add_argument("--k", type=int, required=True, help="test samples per class")
    parser.add_argument("--arch", required=True, choices=["mlp", "cnn"])
    parser.add_argument("--run_type", required=True, choices=["seed", "optim"])
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_csv", required=True, help="shared CSV to append summary rows to")
    parser.add_argument("--step_size", type=float, default=0.01,
                         help="ray-walk step size (paper default 0.01; coarser value trades GI resolution for speed)")
    parser.add_argument("--max_eps", type=float, default=3.0,
                         help="ray-walk search range (paper default 3.0). NOTE: lowering this does NOT change the "
                              "RC metric (%% GI>1) as long as it stays > 1.0 -- capped samples still count as "
                              "covered. It only makes 'mean_gi' an underestimate for samples that would have "
                              "exceeded the new cap, since their GI is reported as the cap value instead.")
    args = parser.parse_args()

    points = [int(p) for p in args.points.split(",")]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        import yaml
        cfg_peek = yaml.safe_load(f)
    val_split = cfg_peek["dataset"].get("val_split", 0.2)

    train_dataset, test_dataset = load_fixed_eval_split(args.data_path, val_split)
    test_indices = sample_test_indices(test_dataset, args.k)
    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=10)

    summary_rows = []
    for point in points:
        ckpt_path = resolve_checkpoint_for_point(args.checkpoint_dir, point)
        model, _ = load_model_from_config(args.config, ckpt_path, device)
        model.eval()

        df = evaluate_checkpoint(model, train_dataset, test_dataset, test_indices, train_indices_by_class, device,
                                  args.step_size, args.max_eps)
        per_sample_path = os.path.join(args.output_dir, f"point_{point}_samples.csv")
        df.to_csv(per_sample_path, index=False)

        row = {
            "arch": args.arch, "run_type": args.run_type, "run_id": args.run_id,
            "point": point, "n_samples": len(df),
            "mean_gi": float(df["mean_gi"].mean()),
            "mean_rc": float(df["coverage"].mean()),
            "median_rc": float(df["coverage"].median()),
            "std_rc": float(df["coverage"].std()),
        }
        summary_rows.append(row)
        print(f"[{args.arch}/{args.run_type}_{args.run_id}] point={point} ckpt={os.path.basename(ckpt_path)} "
              f"step_size={args.step_size} max_eps={args.max_eps} "
              f"mean_gi={row['mean_gi']:.3f} mean_rc={row['mean_rc']:.2f}%")

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    header = not os.path.exists(args.summary_csv)
    summary_df.to_csv(args.summary_csv, mode="a", index=False, header=header)
    print(f"Appended {len(summary_df)} rows -> {args.summary_csv}")


if __name__ == "__main__":
    main()
