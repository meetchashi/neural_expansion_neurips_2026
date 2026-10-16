import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

import torch
import torch.nn.functional as F
import yaml
import importlib
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BASE = (
    "experiments/results/"
    "temp_all_classes/mnist_original_mlp/GI_all_classes_experiment_outputs_plainnn"
)
DEFAULT_CSV     = os.path.join(_BASE, "test_to_all_classes_gi_coverage.csv")
DEFAULT_JSON    = os.path.join(_BASE, "misclassified_train_ids.json")
DEFAULT_DATA    = "data/MNIST-Original"
DEFAULT_OUT_DIR = os.path.join(_BASE, "analysis")
DEFAULT_CONFIG  = None
DEFAULT_CKPT    = None


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
                mapping[int(col[len(prefix):])] = col
            except ValueError:
                pass
    return mapping


def load_mnist_arrays(data_path):
    """Load full train (60k) and test (10k) as flat numpy arrays."""
    transform = transforms.ToTensor()
    full_train_ds = MNIST(root=data_path, train=True,  transform=transform, download=True)
    test_ds       = MNIST(root=data_path, train=False, transform=transform, download=True)

    print("Loading full train set into memory …")
    train_imgs   = np.stack([full_train_ds[i][0].numpy().flatten()
                             for i in tqdm(range(len(full_train_ds)))])
    train_labels = np.array([full_train_ds[i][1] for i in range(len(full_train_ds))])

    print("Loading test set into memory …")
    test_imgs    = np.stack([test_ds[i][0].numpy().flatten()
                             for i in tqdm(range(len(test_ds)))])
    test_labels  = np.array([test_ds[i][1] for i in range(len(test_ds))])

    return train_imgs, train_labels, test_imgs, test_labels


def load_model_from_config(config_path, checkpoint_path, device):
    """Load model based on configuration file (copied from experiment script)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    module_path  = model_config.get('module', 'src.models')
    model_file   = model_config.get('file', model_config['name'].lower())
    try:
        module      = importlib.import_module(f"{module_path}.{model_file}")
        model_class = getattr(module, model_config['name'])
        model       = model_class(**model_config.get('args', {})).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, config
    except (ImportError, AttributeError) as e:
        print(f"Error loading model: {e}")
        raise


# ---------------------------------------------------------------------------
# Q5 – Misclassified test sample neighbor analysis
# ---------------------------------------------------------------------------

def misclassified_test_neighbor_analysis(df, train_imgs, test_imgs, output_dir):
    print("\n=== Q5: Misclassified Test Sample Neighbor Analysis ===")

    mis = df[df['true_label'] != df['predicted_class']]
    if mis.empty:
        print("No misclassified samples.")
        return

    df_cols = df.columns.tolist()
    all_rows = []

    for _, row in mis.iterrows():
        sid      = int(row['sample_id'])
        true_c   = int(row['true_label'])
        pred_c   = int(row['predicted_class'])
        test_vec = test_imgs[sid].reshape(1, -1)

        print(f"\n  sample_id={sid}  true={true_c}  pred={pred_c}")

        for neighbor_class, label in [(true_c, 'true_class'), (pred_c, 'pred_class')]:
            col_map = gi_cols_for_class(df_cols, neighbor_class)
            if not col_map:
                continue

            gi_items = [(gidx, float(row[col]))
                        for gidx, col in col_map.items()
                        if not pd.isna(row[col])]
            if not gi_items:
                continue

            # Sort by highest GI
            gi_items_sorted_by_gi = sorted(gi_items, key=lambda x: x[1], reverse=True)
            top5_by_gi = gi_items_sorted_by_gi[:5]

            # Sort by lowest L2
            gi_indices = [x[0] for x in gi_items]
            train_vecs = train_imgs[gi_indices]
            l2_dists   = euclidean_distances(test_vec, train_vecs)[0]
            cos_sims   = cosine_similarity(test_vec, train_vecs)[0]

            idx_to_l2  = {gidx: l2_dists[i]  for i, gidx in enumerate(gi_indices)}
            idx_to_cos = {gidx: cos_sims[i]   for i, gidx in enumerate(gi_indices)}

            top5_by_l2 = sorted(gi_items, key=lambda x: idx_to_l2[x[0]])[:5]

            print(f"\n    {label.upper()} — Top 5 by GI:")
            print(f"    {'Rank':>5} {'TrainIdx':>10} {'GI':>8} {'L2':>8} {'Cosine':>8}")
            for rank, (gidx, gval) in enumerate(top5_by_gi, 1):
                print(f"    {rank:>5} {gidx:>10} {gval:>8.4f} "
                      f"{idx_to_l2[gidx]:>8.4f} {idx_to_cos[gidx]:>8.4f}")

            print(f"\n    {label.upper()} — Top 5 by L2 (nearest):")
            print(f"    {'Rank':>5} {'TrainIdx':>10} {'GI':>8} {'L2':>8} {'Cosine':>8}")
            for rank, (gidx, gval) in enumerate(top5_by_l2, 1):
                print(f"    {rank:>5} {gidx:>10} {gval:>8.4f} "
                      f"{idx_to_l2[gidx]:>8.4f} {idx_to_cos[gidx]:>8.4f}")

            mean_gi_top5 = np.mean([x[1] for x in top5_by_gi])
            mean_l2_top5 = np.mean([idx_to_l2[x[0]] for x in top5_by_gi])
            print(f"    Mean GI (top5 by GI) = {mean_gi_top5:.4f}  |  "
                  f"Mean L2 (top5 by GI) = {mean_l2_top5:.4f}")

            for rank, (gidx, gval) in enumerate(top5_by_gi, 1):
                all_rows.append({
                    'test_sample_id':   sid,
                    'true_label':       true_c,
                    'pred_class':       pred_c,
                    'neighbor_class':   neighbor_class,
                    'selection':        'top_gi',
                    'rank_by_gi':       rank,
                    'global_train_idx': gidx,
                    'gi_value':         gval,
                    'l2_distance':      round(idx_to_l2[gidx], 6),
                    'cosine_similarity':round(idx_to_cos[gidx], 6),
                })

    out_df   = pd.DataFrame(all_rows)
    out_path = os.path.join(output_dir, 'q5_misclassified_neighbor_analysis.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Q6 – GI distribution for correctly classified samples
# ---------------------------------------------------------------------------

def gi_distribution_correctly_classified(df, output_dir):
    print("\n=== Q6: GI Distribution for Correctly Classified Samples ===")

    correct = df[(df['true_label'] == df['predicted_class'])].copy()
    df_cols = df.columns.tolist()

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    stat_rows = []

    for c in range(10):
        ax  = axes[c]
        sub = correct[correct['true_label'] == c]
        if sub.empty:
            ax.set_title(f"Class {c} — no data")
            continue

        col_map   = gi_cols_for_class(df_cols, c)
        gi_cols_c = list(col_map.values())

        all_gi_vals = sub[gi_cols_c].values.flatten()
        all_gi_vals = all_gi_vals[~np.isnan(all_gi_vals)]

        # Plot histogram
        bins = np.linspace(0, max(4.0, all_gi_vals.max() + 0.1), 60)
        ax.hist(all_gi_vals, bins=bins, color=colors[c], alpha=0.75, edgecolor='none')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='GI=1')
        ax.set_title(f"Class {c}  (n={len(sub)} test)", fontsize=9)
        ax.set_xlabel("GI value", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        # Per-test-sample stats
        for _, row in sub.iterrows():
            gi_vals = row[gi_cols_c].dropna().values.astype(float)
            if len(gi_vals) == 0:
                continue
            stat_rows.append({
                'sample_id':   int(row['sample_id']),
                'true_label':  c,
                'mean_gi':     round(float(np.mean(gi_vals)), 6),
                'median_gi':   round(float(np.median(gi_vals)), 6),
                'std_gi':      round(float(np.std(gi_vals)), 6),
                'var_gi':      round(float(np.var(gi_vals)), 6),
                'pct_above_1': round(float(100 * np.sum(gi_vals > 1.0) / len(gi_vals)), 4),
            })

    fig.suptitle("GI Distribution — Correctly Classified Test Samples per Class", fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(output_dir, 'q6_gi_distribution_correct_per_class.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_png}")

    stat_df  = pd.DataFrame(stat_rows)
    out_csv  = os.path.join(output_dir, 'q6_gi_stats_correct.csv')
    stat_df.to_csv(out_csv, index=False)
    print(f"Saved stats → {out_csv}")


# ---------------------------------------------------------------------------
# Q7 – GI distribution: misclassified vs correctly classified
# ---------------------------------------------------------------------------

def gi_distribution_misclassified_vs_correct(df, output_dir):
    print("\n=== Q7: GI Distribution — Misclassified vs Correct (by predicted class) ===")

    mis     = df[df['true_label'] != df['predicted_class']]
    correct = df[df['true_label'] == df['predicted_class']]
    df_cols = df.columns.tolist()

    pred_classes_in_mis = mis['predicted_class'].unique()
    n = len(pred_classes_in_mis)
    if n == 0:
        print("No misclassified samples.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    stat_rows = []

    for ax_idx, c_pred in enumerate(sorted(pred_classes_in_mis)):
        ax = axes[ax_idx]
        col_map   = gi_cols_for_class(df_cols, int(c_pred))
        gi_cols_c = list(col_map.values())

        mis_sub     = mis[mis['predicted_class'] == c_pred]
        correct_sub = correct[correct['true_label'] == c_pred]

        def extract_gi(sub):
            vals = sub[gi_cols_c].values.flatten()
            return vals[~np.isnan(vals)].astype(float)

        mis_gi     = extract_gi(mis_sub)
        correct_gi = extract_gi(correct_sub)

        bins = np.linspace(0, max(4.0,
                                  (mis_gi.max() if len(mis_gi) else 0),
                                  (correct_gi.max() if len(correct_gi) else 0)) + 0.1, 60)

        if len(correct_gi) > 0:
            ax.hist(correct_gi, bins=bins, alpha=0.6, color='steelblue', density=True,
                    label=f"Correct (n={len(correct_sub)}) μ={correct_gi.mean():.3f} σ={correct_gi.std():.3f}")
        if len(mis_gi) > 0:
            ax.hist(mis_gi, bins=bins, alpha=0.6, color='tomato', density=True,
                    label=f"Misclassified (n={len(mis_sub)}) μ={mis_gi.mean():.3f} σ={mis_gi.std():.3f}")

        ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, label='GI=1')
        ax.set_title(f"Predicted Class {int(c_pred)}", fontsize=11)
        ax.set_xlabel("GI value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        for group, gi_vals in [('correct', correct_gi), ('misclassified', mis_gi)]:
            if len(gi_vals) == 0:
                continue
            stat_rows.append({
                'class':       int(c_pred),
                'group':       group,
                'mean_gi':     round(float(gi_vals.mean()), 6),
                'median_gi':   round(float(np.median(gi_vals)), 6),
                'std_gi':      round(float(gi_vals.std()), 6),
                'pct_above_1': round(float(100 * np.sum(gi_vals > 1.0) / len(gi_vals)), 4),
            })

    fig.suptitle("GI Distribution: Misclassified vs Correctly Classified\n(towards predicted class training samples)", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(output_dir, 'q7_gi_distribution_misclassified_vs_correct.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_png}")

    stat_df  = pd.DataFrame(stat_rows)
    out_csv  = os.path.join(output_dir, 'q7_gi_comparison_stats.csv')
    stat_df.to_csv(out_csv, index=False)
    print(f"Saved stats → {out_csv}")


# ---------------------------------------------------------------------------
# Q8 – GI stats vs softmax probability
# ---------------------------------------------------------------------------

def gi_stats_vs_class_probability(df, config_path, checkpoint_path, data_path, output_dir):
    print("\n=== Q8: GI Stats vs Softmax Probability ===")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, config = load_model_from_config(config_path, checkpoint_path, device)

    dataset_config = config['dataset']
    transform_cfg  = dataset_config.get('transform', {})
    transform_list = [transforms.ToTensor()]
    if 'normalize' in transform_cfg:
        transform_list.append(
            transforms.Normalize(transform_cfg['normalize']['mean'],
                                 transform_cfg['normalize']['std']))
    transform  = transforms.Compose(transform_list)
    test_ds    = MNIST(root=data_path, train=False, transform=transform, download=True)

    df_cols = df.columns.tolist()

    print("Running inference on test samples …")
    softmax_data = {}  # sample_id -> np array (10,)
    sample_ids = df['sample_id'].astype(int).tolist()
    with torch.no_grad():
        for sid in tqdm(sample_ids):
            img = test_ds[sid][0].unsqueeze(0).to(device)
            logits = model(img)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            softmax_data[sid] = probs

    full_rows   = []
    corr_pairs  = []   # (pct_above_1_pred_class, softmax_prob_pred_class) per test sample

    for _, row in df.iterrows():
        sid      = int(row['sample_id'])
        true_lbl = int(row['true_label'])
        pred_cls = int(row['predicted_class'])
        probs    = softmax_data[sid]

        row_data = {
            'sample_id':       sid,
            'true_label':      true_lbl,
            'predicted_class': pred_cls,
        }
        for c in range(10):
            col_map  = gi_cols_for_class(df_cols, c)
            gi_vals  = row[[v for v in col_map.values()]].dropna().values.astype(float)
            mean_gi  = float(np.mean(gi_vals)) if len(gi_vals) > 0 else 0.0
            pct_gi   = float(100 * np.sum(gi_vals > 1.0) / len(gi_vals)) if len(gi_vals) > 0 else 0.0
            row_data[f'mean_gi_c{c}']      = round(mean_gi, 6)
            row_data[f'pct_above_1_c{c}']  = round(pct_gi, 4)
            row_data[f'softmax_prob_c{c}'] = round(float(probs[c]), 6)

        full_rows.append(row_data)
        corr_pairs.append((row_data[f'pct_above_1_c{pred_cls}'],
                           row_data[f'softmax_prob_c{pred_cls}']))

    full_df  = pd.DataFrame(full_rows)
    out_csv  = os.path.join(output_dir, 'q8_gi_vs_softmax.csv')
    full_df.to_csv(out_csv, index=False)
    print(f"Saved full data → {out_csv}")

    pct_vals   = np.array([x[0] for x in corr_pairs])
    prob_vals  = np.array([x[1] for x in corr_pairs])
    r, p       = scipy_stats.pearsonr(pct_vals, prob_vals)
    print(f"\nPearson r (pct_above_1_pred_class vs softmax_pred_class): {r:.4f}  p={p:.4e}")
    if abs(r) > 0.7:
        print("Strong correlation: GI coverage % tightly tracks model confidence.")
    elif abs(r) > 0.3:
        print("Moderate correlation.")
    else:
        print("Weak correlation.")

    # Scatter: correct vs misclassified
    correct_mask = full_df['true_label'] == full_df['predicted_class']
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mask, title in [
        (axes[0], correct_mask,  "Correctly Classified"),
        (axes[1], ~correct_mask, "Misclassified"),
    ]:
        sub = full_df[mask]
        xs = [sub[f'pct_above_1_c{int(r2["predicted_class"])}'].values[0]
              for _, r2 in sub.iterrows()]
        ys = [sub[f'softmax_prob_c{int(r2["predicted_class"])}'].values[0]
              for _, r2 in sub.iterrows()]
        ax.scatter(xs, ys, alpha=0.7, s=60)
        ax.set_xlabel("GI Coverage % for Predicted Class", fontsize=10)
        ax.set_ylabel("Softmax Probability for Predicted Class", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        if len(xs) > 1:
            r2, _ = scipy_stats.pearsonr(xs, ys)
            ax.annotate(f"r={r2:.3f}", xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10)

    fig.suptitle("GI Coverage % vs Softmax Probability (Predicted Class)", fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(output_dir, 'q8_gi_vs_softmax_probability.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GI Misclassified Deep Analysis (Q5-Q8)")
    parser.add_argument('--csv_path',   type=str, default=DEFAULT_CSV)
    parser.add_argument('--json_path',  type=str, default=DEFAULT_JSON)
    parser.add_argument('--data_path',  type=str, default=DEFAULT_DATA)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--config',     type=str, default=DEFAULT_CONFIG,
                        help='Model config YAML (required for Q8)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT,
                        help='Model checkpoint .pth (required for Q8)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CSV …")
    df = pd.read_csv(args.csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    with open(args.json_path) as f:
        mis_ids = json.load(f)

    train_imgs, train_labels, test_imgs, test_labels = load_mnist_arrays(args.data_path)

    misclassified_test_neighbor_analysis(df, train_imgs, test_imgs, args.output_dir)
    gi_distribution_correctly_classified(df, args.output_dir)
    gi_distribution_misclassified_vs_correct(df, args.output_dir)

    if args.config and args.checkpoint:
        gi_stats_vs_class_probability(df, args.config, args.checkpoint,
                                      args.data_path, args.output_dir)
    else:
        print("\nQ8 skipped (--config and --checkpoint not provided).")
        print("To run Q8: add --config ./configs/training/mnist_mlp.yaml "
              "--checkpoint ./checkpoints/mnist_original_mlp/best_model_mnist.pth")

    print("\nAll questions answered.")


if __name__ == '__main__':
    main()
