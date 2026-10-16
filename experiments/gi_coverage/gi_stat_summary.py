import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

def compute_stats_and_plot(root_folder):
    root = Path(root_folder)

    correct_class_data = {}
    misclassified_data = {}

    for file in root.glob("*class_*.csv"):
        try:
            df = pd.read_csv(file)
            if "percent_gi_above_1" not in df.columns:
                continue
            
            parts = file.stem.split("_")
            if file.name.startswith("misclassified"):
                # misclassified_to_class_{wrong_class}_from_class_{source_class}
                source_class = parts[-1]
                misclassified_data.setdefault(source_class, []).append(df["percent_gi_above_1"])
            else:
                # from_train_to_test_GIs_class_{class_id}
                has_k_component = re.match(r'k\d+$', parts[-1])
                true_class = parts[-2] if has_k_component else parts[-1]
                correct_class_data.setdefault(true_class, []).append(df["percent_gi_above_1"])
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

    # ----------- Boxplot for correct classifications -----------
    if correct_class_data:
        fig, ax = plt.subplots(figsize=(12,6))
        class_labels = sorted(correct_class_data.keys(), key=int)
        data = [pd.concat(correct_class_data[c]) for c in class_labels]

        ax.boxplot(data, labels=class_labels)
        ax.set_title("GI% Above 1 - Correct Classifications")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("% GI > 1")
        plt.tight_layout()
        plt.savefig(root / f"{args.prefix}_boxplot_correct_classifications.png")
        plt.close()
        print("Saved boxplot: boxplot_correct_classifications.png")

        # Save summary statistics for correct classifications
        correct_stats = {
            c: pd.concat(correct_class_data[c]).describe()
            for c in class_labels
        }
        correct_summary = pd.DataFrame(correct_stats).T
        correct_summary.index.name = 'class_id'
        correct_summary.to_csv(root / f"{args.prefix}_correct_classified.csv")
        print("Saved summary: correct_classified.csv")


    # ----------- Boxplot for misclassifications -----------
    if misclassified_data:
        fig, ax = plt.subplots(figsize=(12,6))
        source_labels = sorted(misclassified_data.keys(), key=int)
        data = [pd.concat(misclassified_data[c]) for c in source_labels]

        ax.boxplot(data, labels=source_labels)
        ax.set_title("GI% Above 1 - Misclassifications")
        ax.set_xlabel("Source Class ID")
        ax.set_ylabel("% GI > 1")
        plt.tight_layout()
        plt.savefig(root / f"{args.prefix}_boxplot_misclassifications.png")
        plt.close()
        print("Saved boxplot: boxplot_misclassifications.png")

        # Save summary statistics for misclassifications
        misclassified_stats = {
            c: pd.concat(misclassified_data[c]).describe()
            for c in source_labels
        }
        misclassified_summary = pd.DataFrame(misclassified_stats).T
        misclassified_summary.index.name = 'source_class_id'
        misclassified_summary.to_csv(root / f"{args.prefix}_misclassified.csv")
        print("Saved summary: misclassified.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", type=str, help="Path to folder with CSV files")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix for output file names")

    args = parser.parse_args()

    compute_stats_and_plot(args.root_folder)

    # Example command to run the script:
