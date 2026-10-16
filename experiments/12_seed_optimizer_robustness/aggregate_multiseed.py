"""
Aggregate the 16-run multi-seed / multi-optimizer sweep into growth-curve plots,
final-epoch boxplots, a variance table, and results.md.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCH_LABELS = {"mlp": "MNIST MLP", "cnn": "MNIST CNN"}
FACTOR_LABELS = {"seed": "Random seed", "optim": "Optimizer"}


def plot_growth_curve(summary_df, arch, run_type, out_path):
    sub = summary_df[(summary_df.arch == arch) & (summary_df.run_type == run_type)]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    pivot = sub.pivot(index="point", columns="run_id", values="mean_rc")
    for run_id in pivot.columns:
        ax.plot(pivot.index, pivot[run_id], marker="o", alpha=0.6, label=str(run_id), linewidth=1.2)

    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1)
    ax.plot(mean_curve.index, mean_curve.values, color="black", linewidth=2.5, label="mean")
    ax.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve,
                     color="black", alpha=0.15, label="±1 std")

    ax.set_xlabel("Epochs trained")
    ax.set_ylabel("Ray Coverage (% GI > 1), same-class")
    ax.set_title(f"{ARCH_LABELS[arch]} — RC growth across {FACTOR_LABELS[run_type]}s")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_final_boxplot(results_dir, summary_df, arch, run_type, out_path):
    sub = summary_df[(summary_df.arch == arch) & (summary_df.run_type == run_type)]
    final_point = sub["point"].max()
    run_ids = sorted(sub.run_id.unique(), key=str)

    data, labels = [], []
    for run_id in run_ids:
        per_sample_path = Path(results_dir) / f"{arch}_{run_type}" / str(run_id) / f"point_{final_point}_samples.csv"
        df = pd.read_csv(per_sample_path)
        data.append(df["coverage"].values)
        labels.append(str(run_id))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch in box["boxes"]:
        patch.set_facecolor("#A7C7E7")
    ax.set_ylabel("Ray Coverage (% GI > 1), same-class")
    ax.set_title(f"{ARCH_LABELS[arch]} — final-epoch RC per {FACTOR_LABELS[run_type].lower()}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    return final_point


def significance_vs_untrained(results_dir, arch, run_type, run_id, final_point):
    base = Path(results_dir) / f"{arch}_{run_type}" / str(run_id)
    untrained = pd.read_csv(base / "point_0_samples.csv")["coverage"].values
    trained = pd.read_csv(base / f"point_{final_point}_samples.csv")["coverage"].values
    stat, p = mannwhitneyu(untrained, trained, alternative="two-sided")
    return float(stat), float(p)


def variance_table(summary_df, arch, run_type, final_point):
    sub = summary_df[(summary_df.arch == arch) & (summary_df.run_type == run_type) & (summary_df.point == final_point)]
    means = sub["mean_rc"].values
    return {
        "arch": arch, "run_type": run_type, "n_runs": len(means),
        "mean_of_run_means": float(np.mean(means)),
        "std_across_runs": float(np.std(means)),
        "cv_pct": float(100 * np.std(means) / np.mean(means)) if np.mean(means) != 0 else float("nan"),
        "min_run_mean": float(np.min(means)),
        "max_run_mean": float(np.max(means)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)

    variance_rows = []
    significance_rows = []
    combos = [(a, r) for a in ["mlp", "cnn"] for r in ["seed", "optim"]]

    for arch, run_type in combos:
        sub = summary_df[(summary_df.arch == arch) & (summary_df.run_type == run_type)]
        if sub.empty:
            continue

        growth_path = out_dir / f"{arch}_{run_type}_growth_curve"
        plot_growth_curve(summary_df, arch, run_type, growth_path)

        box_path = out_dir / f"{arch}_{run_type}_final_boxplot"
        final_point = plot_final_boxplot(args.results_dir, summary_df, arch, run_type, box_path)

        variance_rows.append(variance_table(summary_df, arch, run_type, final_point))

        for run_id in sorted(sub.run_id.unique(), key=str):
            stat, p = significance_vs_untrained(args.results_dir, arch, run_type, run_id, final_point)
            significance_rows.append({
                "arch": arch, "run_type": run_type, "run_id": run_id,
                "final_point": final_point, "U": stat, "p_value": p,
            })

    variance_df = pd.DataFrame(variance_rows)
    significance_df = pd.DataFrame(significance_rows)
    variance_df.to_csv(out_dir / "variance_table.csv", index=False)
    significance_df.to_csv(out_dir / "significance_vs_untrained.csv", index=False)

    with open(out_dir / "results.md", "w") as f:
        f.write("# Seed and Optimizer Robustness\n\n")
        f.write("Per-sample and per-checkpoint\n")
        f.write("data in `../summary.csv` and `../<arch>_<factor>/<run_id>/point_*_samples.csv`.\n\n")

        f.write("## Variance across runs, at the final evaluated checkpoint\n\n")
        f.write("| Arch | Factor | n runs | mean of run means (RC%) | std across runs | CV% | min | max |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in variance_rows:
            f.write(f"| {ARCH_LABELS[row['arch']]} | {FACTOR_LABELS[row['run_type']]} | {row['n_runs']} | "
                     f"{row['mean_of_run_means']:.2f} | {row['std_across_runs']:.2f} | {row['cv_pct']:.1f} | "
                     f"{row['min_run_mean']:.2f} | {row['max_run_mean']:.2f} |\n")

        f.write("\n## Training effect significance, per individual run (untrained vs. final checkpoint)\n\n")
        f.write("Mann-Whitney U on the per-sample same-class RC distribution, point 0 vs. final point.\n\n")
        f.write("| Arch | Factor | Run | U | p-value | significant (p<0.05) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in significance_rows:
            sig = "yes" if row["p_value"] < 0.05 else "NO"
            f.write(f"| {ARCH_LABELS[row['arch']]} | {FACTOR_LABELS[row['run_type']]} | {row['run_id']} | "
                     f"{row['U']:.1f} | {row['p_value']:.3e} | {sig} |\n")

        f.write("\n## Figures\n\n")
        for arch, run_type in combos:
            if summary_df[(summary_df.arch == arch) & (summary_df.run_type == run_type)].empty:
                continue
            f.write(f"- `{arch}_{run_type}_growth_curve.{{png,pdf}}` -- RC vs. epochs trained, "
                     f"{ARCH_LABELS[arch]}, across {FACTOR_LABELS[run_type].lower()}s.\n")
            f.write(f"- `{arch}_{run_type}_final_boxplot.{{png,pdf}}` -- final-epoch per-sample RC "
                     f"distribution, one box per {FACTOR_LABELS[run_type].lower()}.\n")


    print(f"Wrote variance_table.csv, significance_vs_untrained.csv, results.md, figures -> {out_dir}")
    print(variance_df.to_string(index=False))


if __name__ == "__main__":
    main()
