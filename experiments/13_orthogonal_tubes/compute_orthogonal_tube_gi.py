"""Measure transverse prediction stability around covered training-to-test rays.

Probe five positions per ray using random Gram-Schmidt orthogonal directions.
The positive/negative GI routine measures the first sampled class transition.
"""
import argparse
import importlib
import os

os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from src.analysis.gi_calculation import compute_GI

PROBE_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]


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


def orthogonal_directions(d_hat_flat, n_dirs, dim, rng, device, ortho_scale):
    """n_dirs random unit vectors orthogonal to d_hat (Gram-Schmidt), scaled."""
    g = torch.tensor(rng.standard_normal((n_dirs, dim)), dtype=torch.float32, device=device)
    proj = (g @ d_hat_flat).unsqueeze(1) * d_hat_flat.unsqueeze(0)
    u = g - proj
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return u * ortho_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ray_pool", required=True)
    parser.add_argument("--n_directions", type=int, default=20)
    parser.add_argument("--ortho_scale", type=float, default=56.0)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--max_eps", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, config = load_model_from_config(args.config, args.checkpoint, device)
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"].lower().replace("_", "").replace("-", "")
    val_split = dataset_config.get("val_split", 0.2)
    norm = dataset_config.get("transform", {}).get("normalize", {"mean": [0.5], "std": [0.5]})

    train_dataset, test_dataset = load_datasets(
        dataset_name, args.data_path, val_split, norm["mean"], norm["std"]
    )

    ray_pool = pd.read_csv(args.ray_pool)
    rng = np.random.RandomState(args.seed)

    n_probes = len(PROBE_FRACS)
    n_dirs = args.n_directions
    batch = n_probes * n_dirs

    results = []
    for _, ray in ray_pool.iterrows():
        if ray["ray_kind"] != "train2train":
            raise ValueError(f"Expected a train2train ray-pool entry, got {ray['ray_kind']!r}")

        ray_id = int(ray["ray_id"])
        correct_class = int(ray["correct_class"])
        g_raw = float(ray["g_raw"])
        direction_norm = float(ray["direction_norm"])

        # anchor = train image (x1), target = test image (x2) -- matches the
        # walk direction actually used to produce g_raw, see build_ray_pool.py docstring.
        x1_img, _ = train_dataset.dataset[int(ray["anchor_source"])]
        x2_img, _ = test_dataset[int(ray["target_source"])]
        direction = (x2_img - x1_img).to(device)  # raw, unnormalized (matches Exp-1 convention)
        dim = direction.numel()
        d_hat_flat = direction.flatten() / direction_norm
        x1_img = x1_img.to(device)

        # Build the batch: n_probes probe points, each repeated n_dirs times
        p_batch = torch.stack([
            x1_img + frac * g_raw * direction
            for frac in PROBE_FRACS
            for _ in range(n_dirs)
        ])  # [n_probes * n_dirs, C, H, W]

        u = orthogonal_directions(d_hat_flat, batch, dim, rng, device, args.ortho_scale)
        u = u.view(batch, *x1_img.shape)

        correct_class_t = torch.full((batch,), correct_class, device=device, dtype=torch.long)

        pos_eps, neg_eps = compute_GI(
            model,
            x1=p_batch,
            x2=p_batch + u,
            direction_normalized=False,  # already unit-normalized + scaled above
            correct_class=correct_class_t,
            step_size=args.step_size,
            max_eps=args.max_eps,
            device=device,
        )

        for i, frac in enumerate(PROBE_FRACS):
            for j in range(n_dirs):
                k = i * n_dirs + j
                results.append({
                    "ray_id": ray_id,
                    "ray_kind": "train2train",
                    "probe_frac": frac,
                    "direction_idx": j,
                    "pos_eps": float(pos_eps[k]),
                    "neg_eps": float(neg_eps[k]),
                    "total_eps": float(pos_eps[k] + neg_eps[k]),
                    "ortho_scale": args.ortho_scale,
                    "search_max_eps": args.max_eps,
                    "tube_radius_physical": float(pos_eps[k]) * args.ortho_scale,
                    "ray_g_raw": g_raw,
                    "ray_direction_norm": direction_norm,
                    "ray_physical_length": g_raw * direction_norm,
                    "correct_class": correct_class,
                })

        print(f"ray {ray_id}: mean pos_eps={pos_eps.mean():.3f}, "
              f"mean tube_radius_physical={pos_eps.mean()*args.ortho_scale:.2f}, "
              f"ray_physical_length={g_raw*direction_norm:.2f}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} rows ({ray_pool.shape[0]} rays x {n_probes} probes x {n_dirs} dirs) -> {args.output_csv}")


if __name__ == "__main__":
    main()
