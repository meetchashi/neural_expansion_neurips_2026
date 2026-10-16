import os
import json
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BASE = (
    "experiments/results/"
    "temp_all_classes/mnist_original_mlp/GI_all_classes_experiment_outputs_plainnn"
)
DEFAULT_CSV     = os.path.join(_BASE, "test_to_all_classes_gi_coverage.csv")
DEFAULT_JSON    = os.path.join(_BASE, "misclassified_train_ids.json")
DEFAULT_OUT_DIR = os.path.join(_BASE, "analysis")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gi_cols_for_class(df_cols, c):
    """Return {global_train_idx: col_name} for all gi_c{c}_* columns."""
    prefix = f"gi_c{c}_"
    mapping = {}
    for col in df_cols:
        if col.startswith(prefix):
            try:
                idx = int(col[len(prefix):])
                mapping[idx] = col
            except ValueError:
                pass
    return mapping


# ---------------------------------------------------------------------------
# Q1 – Coverage uniformity across predicted class
# ---------------------------------------------------------------------------

def coverage_uniformity(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    print("\n=== Q1: Coverage Uniformity (all samples predicted to class c) ===")

    rows = []
    print(f"\n{'Class':>6} {'N predicted':>12} {'Unique vals':>12} {'Uniform':>9} {'Coverage':>12}")
    print("-" * 58)
    for c in range(10):
        subset = df[df['predicted_class'] == c]
        cov_vals = subset[f'coverage_class_{c}'].values
        n = len(cov_vals)
        if n == 0:
            rows.append({'class': c, 'all_predicted_to_this_class': 0,
                         'unique_coverage_values': 0, 'is_uniform': 'N/A', 'coverage_value': None})
            print(f"{c:>6} {'0':>12} {'—':>12} {'—':>9} {'—':>12}")
            continue
        unique_vals = np.unique(np.round(cov_vals, 6))
        is_uniform = float(np.std(cov_vals)) < 0.001
        cov_display = float(np.mean(cov_vals))
        rows.append({
            'class': c,
            'all_predicted_to_this_class': n,
            'unique_coverage_values': len(unique_vals),
            'is_uniform': 'yes' if is_uniform else 'no',
            'coverage_value': round(cov_display, 6),
        })
        print(f"{c:>6} {n:>12} {len(unique_vals):>12} {'yes' if is_uniform else 'no':>9} {cov_display:>11.4f}%")

    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'q1_coverage_uniformity.csv')
    out.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    return out


# ---------------------------------------------------------------------------
# Q2 – Coverage vs training accuracy
# ---------------------------------------------------------------------------

def coverage_vs_training_accuracy(df: pd.DataFrame, mis_ids: dict,
                                   q1_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    print("\n=== Q2: Coverage vs Training Accuracy ===")

    rows = []
    print(f"\n{'Cls':>4} {'Train total':>12} {'Mis train':>10} "
          f"{'Train acc %':>12} {'GI cov %':>10} {'Diff':>8}")
    print("-" * 62)
    for c in range(10):
        col_map = gi_cols_for_class(df.columns.tolist(), c)
        total_train = len(col_map)
        mis_count   = len(mis_ids.get(str(c), []))
        correct_count = total_train - mis_count
        train_acc = 100.0 * correct_count / total_train if total_train > 0 else 0.0

        q1_row = q1_df[q1_df['class'] == c]
        gi_cov = float(q1_row['coverage_value'].values[0]) if not q1_row.empty and q1_row['coverage_value'].values[0] is not None else None
        diff   = (gi_cov - train_acc) if gi_cov is not None else None

        rows.append({
            'class':                   c,
            'total_train_samples':     total_train,
            'misclassified_train_count': mis_count,
            'training_accuracy_pct':   round(train_acc, 4),
            'gi_coverage_pct':         round(gi_cov, 4) if gi_cov is not None else None,
            'difference':              round(diff, 4)   if diff   is not None else None,
        })
        gi_s   = f"{gi_cov:.4f}%" if gi_cov is not None else "N/A"
        diff_s = f"{diff:+.4f}%"  if diff   is not None else "N/A"
        print(f"{c:>4} {total_train:>12} {mis_count:>10} {train_acc:>11.4f}% {gi_s:>10} {diff_s:>8}")

    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'q2_coverage_vs_accuracy.csv')
    out.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    return out


# ---------------------------------------------------------------------------
# Q3 – Discrepancy interpretation
# ---------------------------------------------------------------------------

def coverage_accuracy_discrepancy_analysis(q2_df: pd.DataFrame, output_dir: str):
    print("\n=== Q3: Discrepancy Interpretation ===")
    lines = []
    lines.append("GI Coverage vs Training Accuracy — Per-Class Interpretation")
    lines.append("=" * 65)

    for _, row in q2_df.iterrows():
        c    = int(row['class'])
        diff = row['difference']
        if diff is None:
            lines.append(f"\nClass {c}: N/A (no data)")
            continue
        diff = float(diff)
        msg = f"Class {c}: coverage minus training accuracy = {diff:+.4f} percentage points."
        print(f"\n  {msg}")
        lines.append(f"\n{msg}")

    path = os.path.join(output_dir, 'q3_interpretation.txt')
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
# Q4 – Who are the non-GI training samples? 2x2 contingency
# ---------------------------------------------------------------------------

def who_are_the_non_gi_training_samples(df: pd.DataFrame, mis_ids: dict, output_dir: str):
    print("\n=== Q4: GI vs Train Accuracy Contingency per Class ===")

    contingency_rows = []
    anomalous = {}

    for c in range(10):
        col_map = gi_cols_for_class(df.columns.tolist(), c)  # {global_idx: col_name}
        if not col_map:
            print(f"\nClass {c}: no raw GI columns found — skipping.")
            continue

        # Representative row: first correctly classified test sample for class c
        correct_rows = df[(df['true_label'] == c) & (df['predicted_class'] == c)]
        if correct_rows.empty:
            # Fall back to any row
            correct_rows = df[df['predicted_class'] == c]
        if correct_rows.empty:
            print(f"\nClass {c}: no test samples predicted to this class — skipping.")
            continue
        rep_row = correct_rows.iloc[0]

        # Extract GI values for this representative row
        low_gi_indices  = []   # global train idx with GI <= 1
        high_gi_indices = []   # global train idx with GI >  1
        for global_idx, col_name in col_map.items():
            gi_val = rep_row[col_name]
            if pd.isna(gi_val):
                continue
            if float(gi_val) > 1.0:
                high_gi_indices.append(global_idx)
            else:
                low_gi_indices.append(global_idx)

        mis_set = set(int(x) for x in mis_ids.get(str(c), []))

        # 2×2 counts
        low_gi_set  = set(low_gi_indices)
        high_gi_set = set(high_gi_indices)

        A = len(high_gi_set - mis_set)          # correct  + high GI
        B = len(low_gi_set  - mis_set)          # correct  + low  GI  ← anomalous
        C = len(high_gi_set & mis_set)          # misclas  + high GI  ← anomalous
        D = len(low_gi_set  & mis_set)          # misclas  + low  GI
        total = A + B + C + D

        correct_but_low_gi        = sorted(low_gi_set  - mis_set)
        misclassified_but_high_gi = sorted(high_gi_set & mis_set)

        anomalous[f"class_{c}"] = {
            "correct_but_low_gi":        correct_but_low_gi,
            "misclassified_but_high_gi": misclassified_but_high_gi,
        }

        pct = lambda n: round(100.0 * n / total, 4) if total > 0 else 0.0

        contingency_rows.append({
            'class': c,
            'high_gi_correct_A':       A,
            'low_gi_correct_B':        B,
            'high_gi_misclassified_C': C,
            'low_gi_misclassified_D':  D,
            'total_train':             total,
            'pct_A': pct(A), 'pct_B': pct(B),
            'pct_C': pct(C), 'pct_D': pct(D),
        })

        print(f"\n  Class {c}  (rep test sample_id={int(rep_row['sample_id'])})")
        print(f"                    | GI > 1  (high) | GI <= 1 (low)  |")
        print(f"  Correct train     |  A={A:6d} ({pct(A):6.2f}%) | B={B:6d} ({pct(B):6.2f}%) |")
        print(f"  Misclassified tr  |  C={C:6d} ({pct(C):6.2f}%) | D={D:6d} ({pct(D):6.2f}%) |")
        print(f"  B (correct, low GI):  {len(correct_but_low_gi)} samples")
        print(f"  C (misclas, high GI): {len(misclassified_but_high_gi)} samples")

    # Save contingency CSV
    cont_df = pd.DataFrame(contingency_rows)
    path_csv = os.path.join(output_dir, 'q4_gi_vs_train_accuracy_contingency.csv')
    cont_df.to_csv(path_csv, index=False)
    print(f"\nSaved contingency table → {path_csv}")

    # Save anomalous ID lists
    path_json = os.path.join(output_dir, 'q4_anomalous_train_samples.json')
    with open(path_json, 'w') as f:
        json.dump(anomalous, f, indent=2)
    print(f"Saved anomalous train sample IDs → {path_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GI Coverage vs Training Accuracy Analysis")
    parser.add_argument('--csv_path',   type=str, default=DEFAULT_CSV)
    parser.add_argument('--json_path',  type=str, default=DEFAULT_JSON)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading CSV …")
    df = pd.read_csv(args.csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    with open(args.json_path) as f:
        mis_ids = json.load(f)
    total_mis = sum(len(v) for v in mis_ids.values())
    print(f"Loaded misclassified train IDs: {total_mis} total across {len(mis_ids)} classes")

    q1_df = coverage_uniformity(df, args.output_dir)
    q2_df = coverage_vs_training_accuracy(df, mis_ids, q1_df, args.output_dir)
    coverage_accuracy_discrepancy_analysis(q2_df, args.output_dir)
    who_are_the_non_gi_training_samples(df, mis_ids, args.output_dir)

    print("\nAll questions answered.")


if __name__ == '__main__':
    main()
