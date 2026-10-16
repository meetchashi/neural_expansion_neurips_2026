"""
Aggregate the 9 (arm x seed) runs from results/full_sweep/*.json into:
  - results/analysis/long_format.csv         (every checkpoint row, all runs)
  - results/analysis/final_epoch_summary.csv (per-arm mean/std at the final epoch)
  - results/analysis/significance_tests.json (pairwise arm comparisons)
  - results/analysis/curves.png / .pdf       (test_acc / mean_gi / RC vs epoch, mean+-std band)
  - results/analysis/mediator_scatter.png    (final GI/RC vs final test_acc, per run)
  - results/analysis/results.md              (numbers written into prose)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SWEEP_DIR = "results/full_sweep"
OUT_DIR = "results/analysis"
ARM_COLORS = {"control": "#4C72B0", "expand": "#55A868", "restrict": "#C44E52"}
ARM_ORDER = ["control", "expand", "restrict"]


def load_long_format():
    rows = []
    for fpath in sorted(glob.glob(os.path.join(SWEEP_DIR, "*.json"))):
        with open(fpath) as f:
            d = json.load(f)
        for r in d["log"]:
            rows.append(r)
    df = pd.DataFrame(rows)
    return df


def plot_curves(df, out_dir):
    metrics = [
        ("test_acc", "Test accuracy (%)"),
        ("mean_gi", "Mean same-class train-to-train GI"),
        ("pct_gi_gt1", "Ray Coverage: % GI > 1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for arm in ARM_ORDER:
            sub = df[df["arm"] == arm]
            g = sub.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(g["epoch"], g["mean"], label=arm, color=ARM_COLORS[arm], marker="o", markersize=3)
            ax.fill_between(g["epoch"], g["mean"] - g["std"], g["mean"] + g["std"],
                             color=ARM_COLORS[arm], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle("Causal intervention on Ray Coverage (Fashion-MNIST CNN, 3 seeds/arm, shaded = ±1 std)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=180)
    fig.savefig(os.path.join(out_dir, "curves.pdf"))
    plt.close(fig)


def plot_mediator_scatter(df, out_dir):
    final_epoch = df["epoch"].max()
    sub = df[df["epoch"] == final_epoch]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, xlabel in zip(
        axes, ["mean_gi", "pct_gi_gt1"], ["Mean GI (final epoch)", "Ray Coverage %GI>1 (final epoch)"]
    ):
        for arm in ARM_ORDER:
            arm_sub = sub[sub["arm"] == arm]
            ax.scatter(arm_sub[metric], arm_sub["test_acc"], color=ARM_COLORS[arm],
                       label=arm, s=70, edgecolor="k", linewidth=0.5, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Mediator (GI/RC) vs. outcome (test accuracy) at final epoch ({final_epoch}), one point per run")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mediator_scatter.png"), dpi=180)
    plt.close(fig)


def pairwise_tests(df, metric, epoch):
    sub = df[df["epoch"] == epoch]
    out = {}
    for a, b in [("expand", "control"), ("restrict", "control"), ("expand", "restrict")]:
        xa = sub[sub["arm"] == a][metric].values
        xb = sub[sub["arm"] == b][metric].values
        t, p_t = stats.ttest_ind(xa, xb, equal_var=False)
        u, p_u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        out[f"{a}_vs_{b}"] = {
            "n_a": len(xa), "n_b": len(xb),
            "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
            "diff": float(np.mean(xa) - np.mean(xb)),
            "welch_t": float(t), "welch_p": float(p_t),
            "mannwhitney_u": float(u), "mannwhitney_p": float(p_u),
            "values_a": xa.tolist(), "values_b": xb.tolist(),
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_long_format()
    df.to_csv(os.path.join(OUT_DIR, "long_format.csv"), index=False)

    final_epoch = df["epoch"].max()
    final = df[df["epoch"] == final_epoch]
    summary = final.groupby("arm")[["test_acc", "train_acc_full", "mean_gi", "median_gi", "pct_gi_gt1"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUT_DIR, "final_epoch_summary.csv"))

    plot_curves(df, OUT_DIR)
    plot_mediator_scatter(df, OUT_DIR)

    tests = {
        "final_epoch": int(final_epoch),
        "test_acc": pairwise_tests(df, "test_acc", final_epoch),
        "mean_gi": pairwise_tests(df, "mean_gi", final_epoch),
        "pct_gi_gt1": pairwise_tests(df, "pct_gi_gt1", final_epoch),
    }
    with open(os.path.join(OUT_DIR, "significance_tests.json"), "w") as f:
        json.dump(tests, f, indent=2)

    # ---- results.md ----
    lines = []
    lines.append("# Causal intervention on Ray Coverage — results (Fashion-MNIST CNN, 3 seeds x 3 arms)\n")
    lines.append(f"Final epoch measured: {final_epoch}. 3 seeds per arm (42, 43, 44).\n")
    lines.append("## Final-epoch summary (mean ± std over 3 seeds)\n")
    lines.append("| Arm | Test acc (%) | Train acc (%) | Mean GI | Ray Coverage (%GI>1) |")
    lines.append("|---|---|---|---|---|")
    for arm in ARM_ORDER:
        sub = final[final["arm"] == arm]
        lines.append(
            f"| {arm} | {sub['test_acc'].mean():.2f} ± {sub['test_acc'].std():.2f} "
            f"| {sub['train_acc_full'].mean():.2f} ± {sub['train_acc_full'].std():.2f} "
            f"| {sub['mean_gi'].mean():.3f} ± {sub['mean_gi'].std():.3f} "
            f"| {sub['pct_gi_gt1'].mean():.2f} ± {sub['pct_gi_gt1'].std():.2f} |"
        )
    lines.append("")
    lines.append("## Pairwise comparisons at the final epoch (n=3 seeds/arm)\n")
    for metric_name, metric_key in [("Test accuracy", "test_acc"), ("Mean GI", "mean_gi"), ("Ray Coverage", "pct_gi_gt1")]:
        lines.append(f"**{metric_name}**")
        lines.append("| Comparison | mean(A) | mean(B) | diff (A-B) | Welch t p | Mann-Whitney p |")
        lines.append("|---|---|---|---|---|---|")
        for comp, d in tests[metric_key].items():
            lines.append(
                f"| {comp} | {d['mean_a']:.3f} | {d['mean_b']:.3f} | {d['diff']:+.3f} "
                f"| {d['welch_p']:.4f} | {d['mannwhitney_p']:.4f} |"
            )
        lines.append("")
    lines.append("## Per-seed final-epoch raw numbers\n")
    lines.append("| Arm | Seed | Test acc | Train acc | Mean GI | RC (%GI>1) |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in final.sort_values(["arm", "seed"]).iterrows():
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['test_acc']:.2f} | {row['train_acc_full']:.2f} "
            f"| {row['mean_gi']:.3f} | {row['pct_gi_gt1']:.2f} |"
        )
    lines.append("")
    lines.append("See `curves.png`/`curves.pdf` for full training trajectories (mean ± std band across seeds) ")
    lines.append("and `mediator_scatter.png` for the run-level GI/RC-vs-accuracy relationship at the final epoch.\n")

    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines))

    print("Wrote:", os.path.join(OUT_DIR, "results.md"))
    print(summary)


if __name__ == "__main__":
    main()
