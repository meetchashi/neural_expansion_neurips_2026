"""Generate 16 training configurations for seed and optimizer comparisons.

For each architecture, sweep five SGD seeds and three optimizers at seed 42.
All configurations train for 20 epochs at learning rate 0.001."""
import copy
import os

import yaml

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
EXP_ROOT = os.path.dirname(__file__)

BASE_CONFIGS = {
    "mlp": os.path.join(REPO_ROOT, "configs", "training", "mnist_mlp.yaml"),
    "cnn": os.path.join(REPO_ROOT, "configs", "training", "mnist_cnn.yaml"),
}

SEEDS = [0, 1, 2, 3, 4]
OPTIMIZERS = ["SGD", "Adam", "RMSprop"]
FIXED_OPTIMIZER_SEED = 42  # paper's own default, used for the optimizer sweep
NUM_EPOCHS = 20
SAVE_EVERY = 10_000_000  # larger than any epoch's batch count -> disables mid-epoch saves;
                          # mnist_train.py still always saves at the end of every epoch


def make_config(arch, run_type, run_id, seed, optimizer_name):
    with open(BASE_CONFIGS[arch]) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    cfg["seed"] = seed
    cfg["optimizer"]["name"] = optimizer_name
    # Keep lr identical across optimizers on purpose.
    cfg["optimizer"]["args"] = {"lr": 0.001}

    cfg["training"]["num_epochs"] = NUM_EPOCHS
    cfg["training"]["save_every"] = SAVE_EVERY
    cfg["training"]["checkpoint_dir"] = os.path.join(
        "checkpoints", "seed_optimizer_robustness", arch, f"{run_type}_{run_id}"
    )
    return cfg


def main():
    for arch in BASE_CONFIGS:
        out_dir = os.path.join(EXP_ROOT, "configs", arch)
        os.makedirs(out_dir, exist_ok=True)

        for seed in SEEDS:
            cfg = make_config(arch, "seed", seed, seed=seed, optimizer_name="SGD")
            out_path = os.path.join(out_dir, f"seed_{seed}.yaml")
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"wrote {out_path}")

        for opt in OPTIMIZERS:
            cfg = make_config(
                arch, "optim", opt.lower(), seed=FIXED_OPTIMIZER_SEED, optimizer_name=opt
            )
            out_path = os.path.join(out_dir, f"optim_{opt.lower()}.yaml")
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
