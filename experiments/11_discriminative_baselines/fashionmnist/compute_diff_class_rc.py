"""Measure wrong-class anchor coverage on Fashion-MNIST.

Hold the test prediction fixed and reuse the reference CSV test sample IDs.
Anchors already outside the reference prediction have GI=0."""
import os
os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import importlib

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from src.analysis.gi_calculation import compute_GIs_for_sample


def load_model_from_config(config_path, checkpoint_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = config["model"]
    module_path = model_config.get("module", "src.models")
    model_file = model_config.get("file", model_config["name"].lower())
    module = importlib.import_module(f"{module_path}.{model_file}")
    model_class = getattr(module, model_config["name"])
    model = model_class(**model_config.get("args", {})).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, config


def cluster_train_by_labels(train_dataset, total_class=10):
    clusters = {i: [] for i in range(total_class)}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        clusters[label].append(idx)
    return clusters


def get_image_by_index(dataset, index):
    return dataset[index][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--reference_csv", required=True,
                         help="test_to_all_classes_gi_coverage.csv from the matching all-classes "
                              "run -- reused only for its sample_id list, so the diff-class run "
                              "scores the identical set of test samples.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--n_wrong_classes", type=int, default=3,
                         help="Number of distinct wrong classes to sample per test sample and average over "
                              "(three wrong classes by default).")
    parser.add_argument("--total_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, config = load_model_from_config(args.config, args.checkpoint, device)
    dataset_config = config["dataset"]
    total_classes = dataset_config.get("num_classes", args.total_classes)

    transform_list = [transforms.ToTensor()]
    if "normalize" in dataset_config.get("transform", {}):
        norm = dataset_config["transform"]["normalize"]
        transform_list.append(transforms.Normalize(norm["mean"], norm["std"]))
    transform = transforms.Compose(transform_list)

    full_train = FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)
    val_size = int(dataset_config.get("val_split", 0.2) * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(config.get('seed', 42)))
    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=total_classes)

    ref_df = pd.read_csv(args.reference_csv)
    sample_ids = ref_df["sample_id"].tolist()

    rng = np.random.RandomState(args.seed)

    results = []
    for sample_id in tqdm(sample_ids, desc="Diff-class RC"):
        test_img, true_label = test_dataset[sample_id]
        test_img_dev = test_img.to(device).unsqueeze(0)
        pred = torch.argmax(model(test_img_dev)).item()

        wrong_classes = [c for c in range(total_classes) if c != pred]
        chosen = rng.choice(wrong_classes, size=min(args.n_wrong_classes, len(wrong_classes)), replace=False)

        per_wrong_class_pct = []
        for wc in chosen:
            wc_indices = train_indices_by_class.get(int(wc), [])
            if not wc_indices:
                continue
            wc_imgs = torch.stack([get_image_by_index(train_dataset, i) for i in wc_indices]).to(device)
            # Walk from wrong-class training images toward the test image, checking
            # when the prediction leaves the TEST SAMPLE'S OWN predicted class (pred) --
            # not the wrong class. This is the non-tautological version.
            pos_gis, _ = compute_GIs_for_sample(
                model=model,
                sample=test_img.squeeze(0) if test_img.dim() == 3 else test_img,
                normalized=False,
                class_samples=wc_imgs,
                correct_class=pred,
                step_size=0.01,
                max_eps=3.0,
                device=device,
            )
            per_wrong_class_pct.append(100 * np.sum(pos_gis > 1.0) / len(pos_gis))

        diff_class_rc = float(np.mean(per_wrong_class_pct)) if per_wrong_class_pct else 0.0
        results.append({
            "sample_id": int(sample_id),
            "true_label": int(true_label),
            "predicted_class": int(pred),
            "wrong_classes_used": ",".join(map(str, chosen)),
            "diff_class_rc": diff_class_rc,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} rows -> {args.output_csv}")
    print(out_df["diff_class_rc"].describe())


if __name__ == "__main__":
    main()
