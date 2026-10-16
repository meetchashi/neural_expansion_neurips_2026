import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

def compute_misclassification_stats_by_class_frequency(root_folder, distribution_type, prefix=""):
    root = Path(root_folder)
    
    # Define the class distributions based on the specified type
    if distribution_type.lower() == "lt1":
        train_distribution = {0: 4717, 1: 2866, 2: 2370, 3: 2015, 4: 1806, 
                             5: 1590, 6: 1346, 7: 1034, 8: 886, 9: 730}
    elif distribution_type.lower() == "lt2":
        train_distribution = {0: 4716, 1: 2504, 2: 2026, 3: 1464, 4: 1096, 
                             5: 761, 6: 383, 7: 218, 8: 93, 9: 60}
    else:
        print(f"Unknown distribution type: {distribution_type}")
        return
    
    # Get all values and calculate actual percentiles
    counts = list(train_distribution.values())
    
    # Calculate 33rd and 66th percentiles
    q1 = np.percentile(counts, 33.33)
    q2 = np.percentile(counts, 66.67)
    
    print(f"Distribution type: {distribution_type}")
    print(f"33rd percentile (q1): {q1}")
    print(f"66th percentile (q2): {q2}")
    
    # Map classes to frequency categories based on these percentiles
    frequency_map = {}
    for cls, count in train_distribution.items():
        if count >= q2:
            frequency_map[cls] = "high"
        elif count >= q1:
            frequency_map[cls] = "medium"
        else:
            frequency_map[cls] = "low"
    
    # Print the categorization for reference
    high_classes = [cls for cls, freq in frequency_map.items() if freq == "high"]
    medium_classes = [cls for cls, freq in frequency_map.items() if freq == "medium"]
    low_classes = [cls for cls, freq in frequency_map.items() if freq == "low"]
    
    print(f"High frequency classes: {sorted(high_classes)}")
    print(f"Medium frequency classes: {sorted(medium_classes)}")
    print(f"Low frequency classes: {sorted(low_classes)}")
    
    # Define order for categories - correct logical order
    freq_order = ['high', 'medium', 'low']
    
    # Container for all misclassified data
    all_misclassified_df = []
    
    # Process all misclassified CSV files
    for file in root.glob("misclassified_*.csv"):
        try:
            df = pd.read_csv(file)
            if "percent_gi_above_1" not in df.columns or "predicted_class" not in df.columns or "true_label" not in df.columns:
                print(f"Skipping {file.name}: Missing required columns")
                continue
            
            # Add the file to our collection
            all_misclassified_df.append(df)
            
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")
    
    # Combine all data
    if not all_misclassified_df:
        print("No valid misclassification data found.")
        return
        
    combined_df = pd.concat(all_misclassified_df, ignore_index=True)
    
    # Convert class columns to integers if they're strings
    for col in ['true_label', 'predicted_class']:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].astype(int)
    
    # Add frequency category columns
    combined_df['true_freq'] = combined_df['true_label'].map(lambda x: frequency_map.get(x, "unknown"))
    combined_df['pred_freq'] = combined_df['predicted_class'].map(lambda x: frequency_map.get(x, "unknown"))
    
    # 1. Create a heatmap of misclassification by frequency category
    misclass_by_freq = pd.crosstab(
        combined_df['true_freq'], 
        combined_df['pred_freq'], 
        normalize='index'
    ) * 100
    
    # Reorder rows and columns to ensure high -> medium -> low ordering
    misclass_by_freq = misclass_by_freq.reindex(freq_order, axis=0)
    misclass_by_freq = misclass_by_freq.reindex(freq_order, axis=1)
    
    plt.figure(figsize=(6, 3))
    sns.heatmap(misclass_by_freq, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.xlabel('Predicted Class Frequency')
    plt.ylabel('True Class Frequency')
    plt.title(f'Misclassification Distribution (%) by Class Frequency - {distribution_type.upper()}')
    plt.tight_layout()
    plt.savefig(root / f"{prefix}{distribution_type}_misclass_by_freq.png")
    plt.close()
    print(f"Saved: {prefix}{distribution_type}_misclass_by_freq.png")
    
    # 2. Analyze GI coverage by frequency category
    gi_by_freq = combined_df.groupby(['true_freq', 'pred_freq'])['percent_gi_above_1'].agg(
        ['count', 'mean', 'std', 'min', 'max', 'median']
    ).reset_index()
    
    # Save detailed stats to CSV
    gi_by_freq.to_csv(root / f"{prefix}{distribution_type}_gi_by_freq_stats.csv", index=False)
    print(f"Saved: {prefix}{distribution_type}_gi_by_freq_stats.csv")
    
    # 3. Create a heatmap of mean GI coverage by frequency category
    gi_pivot = gi_by_freq.pivot(index='true_freq', columns='pred_freq', values='mean')
    
    # Reorder rows and columns to ensure high -> medium -> low ordering
    gi_pivot = gi_pivot.reindex(freq_order, axis=0)
    gi_pivot = gi_pivot.reindex(freq_order, axis=1)
    
    plt.figure(figsize=(6, 3))
    sns.heatmap(gi_pivot, annot=True, fmt='.1f', cmap='PuBuGn')
    plt.xlabel('Predicted Class Frequency')
    plt.ylabel('True Class Frequency')
    plt.title(f'Average GI Coverage (% > 1) by Class Frequency - {distribution_type.upper()}')
    plt.tight_layout()
    plt.savefig(root / f"{prefix}{distribution_type}_gi_coverage_by_freq.png")
    plt.close()
    print(f"Saved: {prefix}{distribution_type}_gi_coverage_by_freq.png")
    
    # 4. Create a bar chart comparing GI coverage for different frequency categories
    plt.figure(figsize=(6, 3))
    
    # Convert categorical values to ensure right order for plotting
    gi_by_freq['true_freq'] = pd.Categorical(gi_by_freq['true_freq'], categories=freq_order, ordered=True)
    gi_by_freq['pred_freq'] = pd.Categorical(gi_by_freq['pred_freq'], categories=freq_order, ordered=True)
    gi_by_freq = gi_by_freq.sort_values(['true_freq', 'pred_freq'])
    
    # Create x-positions for grouped bars
    x = np.arange(len(freq_order))
    width = 0.25
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter data for each true_freq and plot
    for i, true_freq in enumerate(freq_order):
        true_freq_data = gi_by_freq[gi_by_freq['true_freq'] == true_freq].set_index('pred_freq').reindex(freq_order).reset_index()
        true_freq_data[['mean', 'std', 'count']] = true_freq_data[['mean', 'std', 'count']].fillna(0)
        positions = x + (i - 1) * width
        bars = ax.bar(positions, true_freq_data['mean'], width, 
                     label=f'True: {true_freq}', 
                     yerr=true_freq_data['std'],
                     capsize=5)
        
        # Add count labels above bars
        for j, bar in enumerate(bars):
            if j < len(true_freq_data):
                count = true_freq_data.iloc[j]['count']
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'n={count}', ha='center', va='bottom', size=9)
    
    ax.set_ylabel('Mean GI Coverage (% > 1)')
    ax.set_title(f'GI Coverage by Class Frequency - {distribution_type.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_order)
    ax.set_xlabel('Predicted Class Frequency')
    ax.legend(title='True Class Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(root / f"{prefix}{distribution_type}_gi_freq_comparison.png")
    plt.close()
    print(f"Saved: {prefix}{distribution_type}_gi_freq_comparison.png")
    
    # 5. Additional visualization: Create a flow diagram showing migration patterns
    # This simulates a Sankey diagram using a heatmap with custom formatting
    flow_counts = pd.crosstab(combined_df['true_freq'], combined_df['pred_freq'])
    
    # Reorder rows and columns to ensure high -> medium -> low ordering
    flow_counts = flow_counts.reindex(freq_order, axis=0)
    flow_counts = flow_counts.reindex(freq_order, axis=1)
    
    flow_pct = flow_counts.div(flow_counts.sum(axis=1), axis=0) * 100
    
    # Create a custom heatmap with arrows to indicate flow
    plt.figure(figsize=(6, 3))
    sns.heatmap(flow_pct, annot=True, fmt='.1f', cmap='YlGnBu', 
                linewidths=1, linecolor='white')
    
    # Add arrows to indicate migration from true class to predicted class
    # This is a simple way to show flow direction
    for i, true_cat in enumerate(flow_pct.index):
        for j, pred_cat in enumerate(flow_pct.columns):
            if flow_pct.iloc[i, j] > 10:  # Only show significant flows
                plt.text(j + 0.5, i + 0.5, '→', 
                         ha='center', va='center', fontsize=20, 
                         color='white' if flow_pct.iloc[i, j] > 50 else 'black')
    
    plt.xlabel('Predicted Class Frequency')
    plt.ylabel('True Class Frequency')
    plt.title(f'Migration Pattern from True to Predicted Class - {distribution_type.upper()}')
    plt.tight_layout()
    plt.savefig(root / f"{prefix}{distribution_type}_class_migration_pattern.png")
    plt.close()
    print(f"Saved: {prefix}{distribution_type}_class_migration_pattern.png")
    
    # 6. Create a summary table of model tendency to misclassify by frequency category
    print("\nMisclassification Analysis Summary:")
    print(f"Distribution type: {distribution_type}")
    print("\nAverage GI Coverage (% > 1) by Frequency Category:")
    print(gi_pivot.to_string())
    
    print("\nSample Migration Patterns (% of true class):")
    print(misclass_by_freq.to_string())
    
    print("\nMisclassification Counts by Frequency Category:")
    print(flow_counts.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", type=str, help="Path to folder with CSV files")
    parser.add_argument("--distribution", type=str, required=True, 
                        choices=["lt1", "lt2"], help="Distribution type (lt1 or lt2)")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix for output file names")

    args = parser.parse_args()

    compute_misclassification_stats_by_class_frequency(args.root_folder, args.distribution, args.prefix)
