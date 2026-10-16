"""Select high-coverage training-to-test rays from an all-class GI CSV.

The anchor is the training image and the target is the test image.
Read global training IDs from gi_c<class>_<index> columns, retain GI > 1,
and record the image-pair L2 distance for tube-radius normalization.
"""
import argparse
import re

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

GI_COL_RE = re.compile(r"^gi_c(\d+)_(\d+)$")


def load_datasets(dataset_name, data_path, val_split, mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_cls = {"mnist": MNIST, "fashionmnist": FashionMNIST}[dataset_name]
    full_train = ds_cls(root=data_path, train=True, transform=transform, download=True)
    test_dataset = ds_cls(root=data_path, train=False, transform=transform, download=True)
    val_size = int(val_split * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gi_csv", required=True,
                         help="Experiment-1 test_to_all_classes_gi_coverage.csv")
    parser.add_argument("--config", required=True,
                         help="Training config YAML (for dataset name/transform/val_split, "
                              "must match the config used to produce --gi_csv)")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--n_rays", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"].lower().replace("_", "").replace("-", "")
    val_split = dataset_config.get("val_split", 0.2)
    norm = dataset_config.get("transform", {}).get("normalize", {"mean": [0.5], "std": [0.5]})

    train_dataset, test_dataset = load_datasets(
        dataset_name, args.data_path, val_split, norm["mean"], norm["std"]
    )

    df = pd.read_csv(args.gi_csv)

    # Collect every (test_idx, train_idx, g) triple whose column is a
    # gi_c{class}_{train_idx} entry AND class == this row's predicted_class
    # Select same-class training-to-test rays.
    # Precompute, once, which columns belong to each class's training pool
    # (all rows share the same column set; avoid re-scanning ~10 x pool-size
    # columns per row).
    cols_by_class = {}
    for col in df.columns:
        m = GI_COL_RE.match(col)
        if m is None:
            continue
        class_id, train_idx = int(m.group(1)), int(m.group(2))
        cols_by_class.setdefault(class_id, []).append((col, train_idx))

    candidates = []
    for _, row in df.iterrows():
        test_idx = int(row["sample_id"])
        pred_class = int(row["predicted_class"])
        for col, train_idx in cols_by_class.get(pred_class, []):
            g = row[col]
            if pd.notna(g) and g > 1.0:
                candidates.append((test_idx, pred_class, train_idx, float(g)))

    print(f"Found {len(candidates)} candidate high-GI (g>1.0) train-to-train rays "
          f"across {len(df)} test samples.")

    rng = np.random.RandomState(args.seed)
    n_rays = min(args.n_rays, len(candidates))
    chosen_idx = rng.choice(len(candidates), size=n_rays, replace=False)
    chosen = [candidates[i] for i in chosen_idx]

    results = []
    for ray_id, (test_idx, pred_class, train_idx, g) in enumerate(chosen):
        test_img, _ = test_dataset[test_idx]
        # train_idx stored in the GI csv is the *global* full_train index
        # (see mnist_test_to_all_classes_gi_coverage.py: global_train_idx =
        # train_dataset.indices[train_ds_idx]), so resolve it via
        # train_dataset.dataset (== full_train), not train_dataset itself.
        train_img, _ = train_dataset.dataset[train_idx]

        direction = (test_img - train_img)  # anchor=train -> target=test, see module docstring
        direction_norm = float(torch.norm(direction.flatten()).item())

        results.append({
            "ray_id": ray_id,
            "ray_kind": "train2train",
            "anchor_source": train_idx,
            "anchor_dataset": "train",
            "target_source": test_idx,
            "target_dataset": "test",
            "rsv_index": "",
            "correct_class": pred_class,
            "g_raw": g,
            "direction_norm": direction_norm,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} rays -> {args.output_csv}")
    print(out_df[["g_raw", "direction_norm"]].describe())


if __name__ == "__main__":
    main()
