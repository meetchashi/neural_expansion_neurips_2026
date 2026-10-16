"""
Untrained/trained and same/different-class ray-coverage controls.

Same-class RC (Cells 1 and 3) comes straight from the coverage_class_{predicted_class}
column of the CSVs produced by
experiments/gi_coverage/test_to_train/mnist_test_to_all_classes_gi_coverage.py
-- that computation already matches the paper's own same-class RC convention
(correct_class = predicted_class = the walked-toward training pool's class).

Different-class RC (Cells 2 and 4) comes from compute_diff_class_rc.py's output,
NOT from the other coverage_class_c columns in the all-classes CSV. Those columns
use correct_class = c (the wrong class), which forces the walk to end exactly at
the test image whose real prediction is never c -- so they are always exactly 0,
by construction, not a measurement. compute_diff_class_rc.py fixes this by keeping
correct_class = predicted_class (the test sample's own class) while walking toward
wrong-class training pools, mirroring the same-class convention. See that script's
docstring for the full explanation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CELL_LABELS = {
    "random_same": "Random,\nsame class",
    "random_diff": "Random,\ndifferent class",
    "trained_same": "Trained,\nsame class",
    "trained_diff": "Trained,\ndifferent class",
}


def same_class_rc(all_classes_csv):
    """Return same-class RC array (one value per test sample) from an
    all-classes coverage CSV: coverage_class_{predicted_class} per row."""
    df = pd.read_csv(all_classes_csv)
    vals = [row[f"coverage_class_{int(row['predicted_class'])}"] for _, row in df.iterrows()]
    return np.array(vals, dtype=float), df["sample_id"].values


def diff_class_rc(diff_csv):
    """Return different-class RC array (one value per test sample) from a
    compute_diff_class_rc.py output CSV."""
    df = pd.read_csv(diff_csv)
    return df["diff_class_rc"].values.astype(float), df["sample_id"].values


def summarize(name, arr):
    return {
        "cell": name,
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_csv", required=True,
                         help="test_to_all_classes_gi_coverage.csv from the RANDOM (untrained) checkpoint run")
    parser.add_argument("--trained_csv", required=True,
                         help="test_to_all_classes_gi_coverage.csv from the TRAINED checkpoint run")
    parser.add_argument("--random_diff_csv", required=True,
                         help="compute_diff_class_rc.py output for the RANDOM checkpoint")
    parser.add_argument("--trained_diff_csv", required=True,
                         help="compute_diff_class_rc.py output for the TRAINED checkpoint")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_same, random_same_ids = same_class_rc(args.random_csv)
    trained_same, trained_same_ids = same_class_rc(args.trained_csv)
    random_diff, random_diff_ids = diff_class_rc(args.random_diff_csv)
    trained_diff, trained_diff_ids = diff_class_rc(args.trained_diff_csv)

    assert (random_same_ids == random_diff_ids).all(), "random same/diff sample_id mismatch"
    assert (trained_same_ids == trained_diff_ids).all(), "trained same/diff sample_id mismatch"

    cells = {
        "random_same": random_same,
        "random_diff": random_diff,
        "trained_same": trained_same,
        "trained_diff": trained_diff,
    }

    # ---------------------------------------------------------------
    # Summary stats
    # ---------------------------------------------------------------
    summary_rows = [summarize(name, arr) for name, arr in cells.items()]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # ---------------------------------------------------------------
    # Significance tests for the three comparisons that matter
    # ---------------------------------------------------------------
    def mwu(a, b):
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        return {"U": float(stat), "p_value": float(p)}

    tests = {
        "random_same_vs_trained_same (training effect)": mwu(cells["random_same"], cells["trained_same"]),
        "trained_same_vs_trained_diff (class specificity)": mwu(cells["trained_same"], cells["trained_diff"]),
        "random_same_vs_random_diff (negative control)": mwu(cells["random_same"], cells["random_diff"]),
        "random_diff_vs_trained_diff (negative control)": mwu(cells["random_diff"], cells["trained_diff"]),
    }
    with open(out_dir / "significance_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    # ---------------------------------------------------------------
    # Boxplot: 4 cells side by side
    # ---------------------------------------------------------------
    order = ["random_same", "random_diff", "trained_same", "trained_diff"]
    data = [cells[k] for k in order]
    labels = [CELL_LABELS[k] for k in order]
    colors = ["#A8D5BA", "#A8D5BA", "#F4A7A7", "#F4A7A7"]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Ray Coverage (% GI > 1)")
    ax.set_title("Ray Coverage — Untrained vs. Trained, Same vs. Different Class")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rc_2x2_boxplot.png", dpi=300)
    plt.savefig(out_dir / "rc_2x2_boxplot.pdf")
    plt.close()

    # ---------------------------------------------------------------
    # Save measured cell summaries and statistical comparisons.
    # ---------------------------------------------------------------
    with open(out_dir / "results.md", "w") as f:
        f.write("# 2x2 Ray Coverage Controls\n\n")
        f.write("| Cell | n | mean RC | median RC | std |\n")
        f.write("|---|---|---|---|---|\n")
        for row in summary_rows:
            f.write(f"| {row['cell']} | {row['n']} | {row['mean']:.2f} | {row['median']:.2f} | {row['std']:.2f} |\n")
        f.write("\n## Significance tests (Mann-Whitney U, two-sided)\n\n")
        for name, res in tests.items():
            f.write(f"- **{name}**: U={res['U']:.1f}, p={res['p_value']:.3e}\n")

    print(f"Saved summary.csv, significance_tests.json, rc_2x2_boxplot.{{png,pdf}}, results.md -> {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
