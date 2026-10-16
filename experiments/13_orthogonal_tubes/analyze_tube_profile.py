"""Summarize orthogonal stability in normalized input-space L2 units.

Outputs include per-model CSVs, tube profiles and a Markdown measurement report.
Report tube_radius / hop_distance separately from
 tube_radius / (g_raw * hop_distance), where g_raw is the ray's measured GI.
These compare transverse stability to the anchor-target distance and to the
entire measured ray segment, respectively. Search settings are saved with the measurements.
"""
import argparse
import os

os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "vs_ray": {
        "ratio_col": "ratio_vs_ray",
        "denom_label": "ray physical length (g_raw x hop distance)",
        "short": "vs. full ray length",
    },
    "vs_hop": {
        "ratio_col": "ratio_vs_hop",
        "denom_label": "single-hop (test<->train) distance",
        "short": "vs. single-hop distance",
    },
}


def summarize_one(csv_path, label, max_eps=16.0):
    df = pd.read_csv(csv_path)
    search_cap = df["search_max_eps"] if "search_max_eps" in df else max_eps
    censoring_rate = float((df["pos_eps"] >= search_cap - 1e-6).mean())

    per_probe = (
        df.groupby(["ray_id", "probe_frac"])
        .agg(tube_radius_physical=("tube_radius_physical", "mean"),
             ray_physical_length=("ray_physical_length", "first"),
             ray_direction_norm=("ray_direction_norm", "first"))
        .reset_index()
    )
    per_probe["ratio_vs_ray"] = per_probe["tube_radius_physical"] / per_probe["ray_physical_length"]
    per_probe["ratio_vs_hop"] = per_probe["tube_radius_physical"] / per_probe["ray_direction_norm"]

    per_ray = (
        per_probe.groupby("ray_id")
        .agg(mean_tube_radius_physical=("tube_radius_physical", "mean"),
             ray_physical_length=("ray_physical_length", "first"),
             ray_direction_norm=("ray_direction_norm", "first"))
        .reset_index()
    )
    per_ray["ratio_vs_ray"] = per_ray["mean_tube_radius_physical"] / per_ray["ray_physical_length"]
    per_ray["ratio_vs_hop"] = per_ray["mean_tube_radius_physical"] / per_ray["ray_direction_norm"]

    profile = per_probe.groupby("probe_frac").agg(
        mean_tube_radius_physical=("tube_radius_physical", "mean"),
        median_ratio_vs_ray=("ratio_vs_ray", "median"),
        q25_vs_ray=("ratio_vs_ray", lambda s: s.quantile(0.25)),
        q75_vs_ray=("ratio_vs_ray", lambda s: s.quantile(0.75)),
        median_ratio_vs_hop=("ratio_vs_hop", "median"),
        q25_vs_hop=("ratio_vs_hop", lambda s: s.quantile(0.25)),
        q75_vs_hop=("ratio_vs_hop", lambda s: s.quantile(0.75)),
    ).reset_index()

    summary = {
        "label": label,
        "n_rays": int(df["ray_id"].nunique()),
        "n_probe_points": int(df["probe_frac"].nunique()),
        "n_directions": int(df.groupby(["ray_id", "probe_frac"])["direction_idx"].count().iloc[0]),
        "censoring_rate": censoring_rate,
        "mean_g_raw": float(df.groupby("ray_id")["ray_g_raw"].first().mean()),
        "mean_ray_physical_length": float(per_ray["ray_physical_length"].mean()),
        "mean_tube_radius_physical": float(per_ray["mean_tube_radius_physical"].mean()),
    }
    for key, m in METRICS.items():
        col = m["ratio_col"]
        summary[f"mean_ratio_{key}"] = float(per_ray[col].mean())
        summary[f"median_ratio_{key}"] = float(per_ray[col].median())
        summary[f"frac_thread_{key}"] = float((per_ray[col] < 0.3).mean())
        summary[f"frac_corridor_or_wider_{key}"] = float((per_ray[col] >= 1.0).mean())
    return summary, profile, per_ray


def interpret(summary, key):
    """Describe measured ratios without asserting a scientific conclusion."""
    return (f"Median ratio {summary[f'median_ratio_{key}']:.3f}; "
            f"mean {summary[f'mean_ratio_{key}']:.3f}; "
            f"{100*summary[f'frac_thread_{key}']:.1f}% below 0.3; "
            f"{100*summary[f'frac_corridor_or_wider_{key}']:.1f}% at or above 1.0.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube_csvs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eps", type=float, default=16.0, help="Search cap for older CSVs without search_max_eps")
    args = parser.parse_args()

    assert len(args.tube_csvs) == len(args.labels)
    os.makedirs(args.output_dir, exist_ok=True)

    summaries = []
    profiles = {}
    per_ray_tables = {}
    for csv_path, label in zip(args.tube_csvs, args.labels):
        summary, profile, per_ray = summarize_one(csv_path, label, args.max_eps)
        summaries.append(summary)
        profiles[label] = profile
        per_ray_tables[label] = per_ray
        per_ray.to_csv(os.path.join(args.output_dir, f"per_ray_{label.replace(' ', '_')}.csv"), index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    # --- Figure: 2 rows (vs_ray, vs_hop) x 3 cols (absolute, ratio profile, histogram) ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    colors = plt.cm.tab10.colors

    for row, key in enumerate(["vs_ray", "vs_hop"]):
        m = METRICS[key]
        med_col, q25_col, q75_col = f"median_ratio_{key}", f"q25_{key}", f"q75_{key}"
        for i, (label, profile) in enumerate(profiles.items()):
            color = colors[i % len(colors)]
            axes[row, 0].plot(profile["probe_frac"], profile["mean_tube_radius_physical"],
                               marker="o", label=label, color=color)
            yerr = np.array([profile[med_col] - profile[q25_col], profile[q75_col] - profile[med_col]])
            axes[row, 1].errorbar(profile["probe_frac"], profile[med_col], yerr=yerr,
                                   marker="o", label=label, color=color, capsize=3)

        axes[row, 0].set_xlabel("Position along ray (fraction of measured ray GI, g)")
        axes[row, 0].set_ylabel("Mean tube radius (physical pixel-L2 units)")
        axes[row, 0].set_title(f"Absolute tube radius vs. position ({m['short']} row)")
        axes[row, 0].legend()

        axes[row, 1].axhline(0.3, color="gray", linestyle="--", linewidth=1, label="corridor threshold (0.3)")
        axes[row, 1].axhline(1.0, color="gray", linestyle=":", linewidth=1, label="1.0")
        axes[row, 1].set_xlabel("Position along ray (fraction of measured ray GI, g)")
        axes[row, 1].set_ylabel(f"Tube radius / {m['denom_label']}\n(median, IQR)")
        axes[row, 1].set_title(f"Tube-to-reference ratio ({m['short']})")
        axes[row, 1].legend(fontsize=8)

        bins = np.logspace(np.log10(0.05), np.log10(30), 25)
        for i, (label, per_ray) in enumerate(per_ray_tables.items()):
            color = colors[i % len(colors)]
            axes[row, 2].hist(per_ray[m["ratio_col"]], bins=bins, alpha=0.5, label=label, color=color)
        axes[row, 2].axvline(0.3, color="gray", linestyle="--", linewidth=1, label="thread/corridor thresholds")
        axes[row, 2].axvline(1.0, color="gray", linestyle="--", linewidth=1)
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_xlabel(f"Per-ray tube radius / {m['denom_label']} (log scale)")
        axes[row, 2].set_ylabel("Number of rays")
        axes[row, 2].set_title(f"Distribution of per-ray ratios ({m['short']})")
        axes[row, 2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "tube_profile.png"), dpi=150)
    fig.savefig(os.path.join(args.output_dir, "tube_profile.pdf"))
    plt.close(fig)

    # --- results.md ---
    lines = ["# Orthogonal Tube Stability\n"]
    lines.append(
        "Two reference distances are reported side by side -- **do not average them "
        "together**, they answer different questions (see script docstring):\n"
        "- **vs. full ray length** = tube radius / (g_raw x hop distance): is the tube as "
        "wide as the *entire* segment the original experiment confirmed stable? Since our "
        "ray pool selects g_raw > 1.0, this denominator "
        "already extends past the training image itself, so it is the *stricter* bar.\n"
        "- **vs. single-hop distance** = tube radius / hop distance: is the tube as wide as "
        "the actual physical distance between the one test image and the one training "
        "image (the paper's own \"GI=1.0\" reference point)? This is the more intuitive, "
        "dataset-grounded comparison.\n"
    )
    lines.append("| Model | n rays | mean g_raw | Median ratio (vs. full ray) | Median ratio (vs. single-hop) | % >=1.0 (vs. full ray) | % >=1.0 (vs. single-hop) |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['label']} | {s['n_rays']} | {s['mean_g_raw']:.2f} | "
            f"{s['median_ratio_vs_ray']:.3f} | {s['median_ratio_vs_hop']:.3f} | "
            f"{s['frac_corridor_or_wider_vs_ray']*100:.0f}% | {s['frac_corridor_or_wider_vs_hop']*100:.0f}% |"
        )
    lines.append("")

    for key in ["vs_ray", "vs_hop"]:
        lines.append(f"## Interpretation -- {METRICS[key]['short']} (descriptive ratio thresholds)\n")
        for s in summaries:
            lines.append(f"**{s['label']}**: {interpret(s, key)}")
            lines.append("")

    lines.append("## Search-cap measurements\n")
    for s in summaries:
        lines.append(
            f"- {s['label']}: {s['censoring_rate']*100:.1f}% of positive-direction "
            "measurements reached the configured search cap. The CSV records "
            "search_max_eps and ortho_scale for new measurements."
        )

    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved summary.csv, tube_profile.{{png,pdf}}, results.md -> {args.output_dir}")


if __name__ == "__main__":
    main()
