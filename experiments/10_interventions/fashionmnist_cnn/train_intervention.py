"""
Fashion-MNIST CNN directional-stability intervention training.

Reuses: src.models.cnn.CNN (paper's exact Fashion-MNIST CNN, configs/training/
fashion_mnist_cnn.yaml: 1->32->64 channels, 28x28 input), src.analysis.gi_calculation
.compute_GI (unmodified GI/ray-walk routine).

Deviations from configs/training/fashion_mnist_cnn.yaml, noted explicitly:
  1. Trains on the full 60k-image train set with no train/val split, same rationale as
     the MLP version (no early stopping used here; removes one source of cross-run
     variability from the GI-eval pair pool).
  2. Full-train-set accuracy at each checkpoint is measured on a fixed 5000-image
     subsample rather than all 60k, since a CNN forward pass is materially more
     expensive than the MLP's and this is evaluated at every epoch.
  3. BatchNorm statistics: the extra forward pass on perturbed inputs (expand/restrict
     arms) runs through the same BatchNorm layers in train() mode as the clean batch,
     so its running statistics get updated by both the clean and the perturbed batch
     each step. This is the standard (simple, non-"dual-BN") way adversarial-style
     perturbation training is usually implemented, and is called out here rather than
     silently applied.
"""
import os
os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.cnn import CNN
from src.analysis.gi_calculation import compute_GI

from regularizers import intervention_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_gi_eval_pairs(train_dataset, num_classes=10, pairs_per_class=20, seed=123):
    """Fixed same-class train-to-train pair pool, independent of training seed."""
    rng = np.random.RandomState(seed)
    by_class = {c: [] for c in range(num_classes)}
    for idx in range(len(train_dataset)):
        by_class[train_dataset.targets[idx].item()].append(idx)

    pairs = []  # (idx1, idx2, label)
    for c in range(num_classes):
        pool = np.array(by_class[c])
        idx1 = rng.choice(pool, size=pairs_per_class, replace=False)
        remaining = np.setdiff1d(pool, idx1)
        idx2 = rng.choice(remaining, size=pairs_per_class, replace=False)
        for a, b in zip(idx1, idx2):
            pairs.append((int(a), int(b), c))
    return pairs


def build_train_acc_subsample(train_dataset, n=5000, seed=321):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_dataset), size=n, replace=False)
    return idx.tolist()


def measure_gi_rc(model, train_dataset, pairs, device, step_size=0.01, max_eps=3.0):
    """Same-class train-to-train GI, matching Appendix B's protocol exactly."""
    model.eval()
    x1 = torch.stack([train_dataset[i][0] for i, _, _ in pairs]).to(device)
    x2 = torch.stack([train_dataset[j][0] for _, j, _ in pairs]).to(device)
    labels = torch.tensor([c for _, _, c in pairs], device=device)
    pos_eps, _ = compute_GI(
        model, x1, x2, direction_normalized=False,
        correct_class=labels, step_size=step_size, max_eps=max_eps, device=device,
    )
    model.train()
    return {
        "mean_gi": float(np.mean(pos_eps)),
        "median_gi": float(np.median(pos_eps)),
        "pct_gi_gt1": float(100.0 * np.mean(pos_eps > 1.0)),
    }


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / total, loss_sum / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["control", "expand", "restrict"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=2.5)
    p.add_argument("--eps_frac", type=float, default=0.08)
    p.add_argument("--lam_warmup_epochs", type=int, default=3,
                    help="Linearly ramp lam from 0 to --lam over this many epochs, "
                         "so CE establishes real class structure before the "
                         "perturbation term applies at full strength.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--gi_checkpoints", type=str, default="0,1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--gi_pairs_per_class", type=int, default=20)
    p.add_argument("--train_acc_subsample", type=int, default=5000)
    p.add_argument("--data_path", type=str, default="./data/FashionMNIST-Original")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    gi_checkpoints = set(int(e) for e in args.gi_checkpoints.split(","))

    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = f"{args.arm}_seed{args.seed}"

    # Data: fixed pools for GI eval pairs / train-acc subsample are seeded
    # independently of args.seed, so they are IDENTICAL across all runs/arms.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)
    gi_pairs = build_gi_eval_pairs(train_dataset, pairs_per_class=args.gi_pairs_per_class, seed=123)
    train_acc_idx = build_train_acc_subsample(train_dataset, n=args.train_acc_subsample, seed=321)
    train_acc_subset = torch.utils.data.Subset(train_dataset, train_acc_idx)

    # Model init + minibatch shuffling ARE seeded by --seed (this is the source of
    # across-run variability we replicate over 3 seeds).
    set_seed(args.seed)
    model = CNN(num_classes=10, input_channels=1, hidden_channels1=32, hidden_channels2=64,
                input_height=28, input_width=28).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_acc_loader = DataLoader(train_acc_subset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    log_rows = []

    def do_checkpoint_eval(epoch):
        test_acc, test_loss = eval_accuracy(model, test_loader, device)
        train_acc, train_loss_sub = eval_accuracy(model, train_acc_loader, device)
        gi_stats = measure_gi_rc(model, train_dataset, gi_pairs, device)
        row = {
            "arm": args.arm, "seed": args.seed, "epoch": epoch,
            "test_acc": test_acc, "test_ce": test_loss,
            "train_acc_full": train_acc, "train_ce_full": train_loss_sub,
            **gi_stats,
        }
        log_rows.append(row)
        print(f"[{run_tag}] epoch={epoch:2d} test_acc={test_acc:6.2f} "
              f"train_acc={train_acc:6.2f} mean_gi={gi_stats['mean_gi']:.3f} "
              f"RC(%GI>1)={gi_stats['pct_gi_gt1']:.2f}", flush=True)
        return row

    t0 = time.time()
    do_checkpoint_eval(0)  # untrained baseline

    for epoch in range(1, args.epochs + 1):
        lam_eff = args.lam * min(1.0, epoch / max(1, args.lam_warmup_epochs))
        running_ce, running_hinge, running_margin, n_batches = 0.0, 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, ce, hinge, m_pert = intervention_loss(
                model, x, y, args.arm, lam_eff, args.tau, args.eps_frac
            )
            loss.backward()
            optimizer.step()
            running_ce += ce.item()
            running_hinge += hinge.item()
            if not torch.isnan(m_pert):
                running_margin += m_pert.item()
            n_batches += 1

        if epoch in gi_checkpoints:
            row = do_checkpoint_eval(epoch)
            row["train_ce_batchavg"] = running_ce / n_batches
            row["train_hinge_batchavg"] = running_hinge / n_batches
            row["train_perturbed_margin_batchavg"] = running_margin / n_batches

    elapsed = time.time() - t0
    print(f"[{run_tag}] done in {elapsed/60:.1f} min")

    out_json = os.path.join(args.output_dir, f"{run_tag}.json")
    with open(out_json, "w") as f:
        json.dump({
            "arm": args.arm, "seed": args.seed, "lam": args.lam, "tau": args.tau,
            "eps_frac": args.eps_frac, "epochs": args.epochs,
            "elapsed_sec": elapsed, "log": log_rows,
        }, f, indent=2)
    print(f"[{run_tag}] saved -> {out_json}")

    ckpt_path = os.path.join(args.output_dir, f"{run_tag}_final.pth")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
