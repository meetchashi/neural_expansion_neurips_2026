import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import glob
import json
from matplotlib.ticker import MaxNLocator

def load_config(config_path):
    """Load the experiment configuration YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_singular_values(sample_dir, checkpoint_name, max_dims=784):
    """Extract singular values from a sample's checkpoint SVD data."""
    try:
        svd_data_path = os.path.join(sample_dir, checkpoint_name, "svd_data.npz")
        
        if not os.path.exists(svd_data_path):
            print(f"Warning: SVD data not found at {svd_data_path}")
            return None
        
        with np.load(svd_data_path) as data:
            singular_values = data['S']
            
            # Pad or truncate to max_dims
            if len(singular_values) < max_dims:
                padded = np.zeros(max_dims)
                padded[:len(singular_values)] = singular_values
                return padded
            else:
                return singular_values[:max_dims]
                
    except Exception as e:
        print(f"Error loading SVD data from {svd_data_path}: {e}")
        return None

def process_checkpoint_data(config):
    """Process SVD data for all checkpoints and create summary statistics."""
    output_dir = os.path.expanduser(config['output_dir'])
    base_dir = output_dir  # Base directory where results are stored
    
    # Get checkpoint names without file extension
    checkpoint_names = [os.path.basename(cp).split('.')[0] for cp in 
                        config['checkpoints']['custom_checkpoints']]
    
    # Determine classes to analyze
    classes = config['dataset'].get('classes', list(range(10)))
    
    # Define maximum dimensions for singular values
    max_dims = config['gi_analysis'].get('num_rsvs', 784)
    
    print(f"Processing {len(checkpoint_names)} checkpoints across {len(classes)} classes...")
    
    # Create summary directory
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Initialize dataframes for results
    columns = [f'sv_{i}' for i in range(max_dims)]
    
    # Create summary dataframes
    summary_all = pd.DataFrame(index=checkpoint_names, columns=columns)
    summary_correct = pd.DataFrame(index=checkpoint_names, columns=columns)
    summary_incorrect = pd.DataFrame(index=checkpoint_names, columns=columns)
    
    # Class-specific summary dataframes
    class_correct_dfs = {c: pd.DataFrame(index=checkpoint_names, columns=columns) for c in classes}
    class_incorrect_dfs = {c: pd.DataFrame(index=checkpoint_names, columns=columns) for c in classes}
    
    # Process each checkpoint
    for checkpoint_name in tqdm(checkpoint_names, desc="Processing checkpoints"):
        # Collect all singular values for this checkpoint
        all_correct_svs = []
        all_incorrect_svs = []
        
        # Class-specific collections
        class_correct_svs = {c: [] for c in classes}
        class_incorrect_svs = {c: [] for c in classes}
        
        # Process each class
        for class_idx in classes:
            # Process correct samples
            correct_dir = os.path.join(base_dir, f"class_{class_idx}", "correct")
            if os.path.exists(correct_dir):
                for sample_dir in glob.glob(os.path.join(correct_dir, "sample_*")):
                    sv = extract_singular_values(sample_dir, checkpoint_name, max_dims)
                    if sv is not None:
                        all_correct_svs.append(sv)
                        class_correct_svs[class_idx].append(sv)
            
            # Process incorrect samples
            incorrect_dir = os.path.join(base_dir, f"class_{class_idx}", "incorrect")
            if os.path.exists(incorrect_dir):
                for sample_dir in glob.glob(os.path.join(incorrect_dir, "sample_*")):
                    sv = extract_singular_values(sample_dir, checkpoint_name, max_dims)
                    if sv is not None:
                        all_incorrect_svs.append(sv)
                        class_incorrect_svs[class_idx].append(sv)
        
        # Compute averages for this checkpoint
        if all_correct_svs:
            avg_correct = np.mean(all_correct_svs, axis=0)
            summary_correct.loc[checkpoint_name] = avg_correct
        
        if all_incorrect_svs:
            avg_incorrect = np.mean(all_incorrect_svs, axis=0)
            summary_incorrect.loc[checkpoint_name] = avg_incorrect
        
        # Compute overall average
        all_svs = all_correct_svs + all_incorrect_svs
        if all_svs:
            avg_all = np.mean(all_svs, axis=0)
            summary_all.loc[checkpoint_name] = avg_all
        
        # Compute class-specific averages
        for class_idx in classes:
            if class_correct_svs[class_idx]:
                avg_class_correct = np.mean(class_correct_svs[class_idx], axis=0)
                class_correct_dfs[class_idx].loc[checkpoint_name] = avg_class_correct
            
            if class_incorrect_svs[class_idx]:
                avg_class_incorrect = np.mean(class_incorrect_svs[class_idx], axis=0)
                class_incorrect_dfs[class_idx].loc[checkpoint_name] = avg_class_incorrect
    
    # Save summary CSVs
    summary_all.to_csv(os.path.join(summary_dir, "average_singular_values_all.csv"))
    summary_correct.to_csv(os.path.join(summary_dir, "average_singular_values_correct.csv"))
    summary_incorrect.to_csv(os.path.join(summary_dir, "average_singular_values_incorrect.csv"))
    
    # Save class-specific CSVs
    for class_idx in classes:
        os.makedirs(os.path.join(summary_dir, f"class_{class_idx}"), exist_ok=True)
        class_correct_dfs[class_idx].to_csv(os.path.join(summary_dir, f"class_{class_idx}", "average_singular_values_correct.csv"))
        class_incorrect_dfs[class_idx].to_csv(os.path.join(summary_dir, f"class_{class_idx}", "average_singular_values_incorrect.csv"))
    
    return summary_all, summary_correct, summary_incorrect, class_correct_dfs, class_incorrect_dfs, summary_dir

def plot_singular_value_evolution(summary_df, output_path, title, num_to_plot=100):
    """Plot the evolution of singular values across checkpoints."""
    plt.figure(figsize=(14, 8))
    
    # Get checkpoint order based on training progression
    checkpoints = summary_df.index
    
    # We'll only plot a subset of singular values to avoid overcrowding
    indices_to_plot = np.arange(0, min(num_to_plot, summary_df.shape[1]))
    
    # Extract epoch and batch from checkpoint names for better x-axis
    def extract_epoch_batch(checkpoint):
        import re
        match = re.search(r"epoch(\d+)_batch(\d+)", checkpoint)
        if match:
            epoch, batch = int(match.group(1)), int(match.group(2))
            # Combine into a floating point value for plotting
            # e.g., epoch 1, batch 200 out of 400 batches would be 1.5
            return epoch + (batch / 400)  # Assuming 400 batches per epoch as in config
        return 0  # Default value if pattern doesn't match
    
    x_values = [extract_epoch_batch(cp) for cp in checkpoints]
    
    # Plot for each singular value index
    for i in indices_to_plot:
        column = f'sv_{i}'
        if column in summary_df.columns:
            values = summary_df[column].values
            if i % 10 == 0:  # Make every 10th line thicker for visibility
                plt.plot(x_values, values, label=f"SV {i}", linewidth=2.0)
            else:
                plt.plot(x_values, values, alpha=0.4, linewidth=0.8)
    
    plt.xlabel('Training Progress (Epoch)')
    plt.ylabel('Singular Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize='small')
    
    # Force x-axis to show integer tick marks for epochs
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_visualizations(summary_all, summary_correct, summary_incorrect, 
                         class_correct_dfs, class_incorrect_dfs, summary_dir):
    """Create visualizations based on the summary data."""
    # Create plots directory
    plots_dir = os.path.join(summary_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot overall summaries
    plot_singular_value_evolution(
        summary_all, 
        os.path.join(plots_dir, "singular_values_evolution_all.png"),
        "Evolution of Average Singular Values (All Samples)"
    )
    
    plot_singular_value_evolution(
        summary_correct, 
        os.path.join(plots_dir, "singular_values_evolution_correct.png"),
        "Evolution of Average Singular Values (Correctly Classified Samples)"
    )
    
    plot_singular_value_evolution(
        summary_incorrect, 
        os.path.join(plots_dir, "singular_values_evolution_incorrect.png"),
        "Evolution of Average Singular Values (Incorrectly Classified Samples)"
    )
    
    # Plot class-specific summaries
    for class_idx, df in class_correct_dfs.items():
        plot_singular_value_evolution(
            df, 
            os.path.join(plots_dir, f"singular_values_evolution_class_{class_idx}_correct.png"),
            f"Evolution of Average Singular Values (Class {class_idx}, Correct)"
        )
    
    for class_idx, df in class_incorrect_dfs.items():
        plot_singular_value_evolution(
            df, 
            os.path.join(plots_dir, f"singular_values_evolution_class_{class_idx}_incorrect.png"),
            f"Evolution of Average Singular Values (Class {class_idx}, Incorrect)"
        )
    
    # Create comparison plots: correct vs incorrect
    plt.figure(figsize=(14, 8))
    
    # Get first 20 singular values for the last checkpoint
    last_checkpoint = summary_correct.index[-1]
    num_to_compare = 20
    
    correct_values = summary_correct.loc[last_checkpoint].iloc[:num_to_compare].values
    incorrect_values = summary_incorrect.loc[last_checkpoint].iloc[:num_to_compare].values
    
    indices = np.arange(num_to_compare)
    width = 0.35
    
    plt.bar(indices - width/2, correct_values, width, label='Correctly Classified')
    plt.bar(indices + width/2, incorrect_values, width, label='Incorrectly Classified')
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Average Magnitude')
    plt.title(f'Comparison of Top {num_to_compare} Singular Values\n(Checkpoint: {last_checkpoint})')
    plt.xticks(indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correct_vs_incorrect_comparison.png"))
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate SVD summary statistics across checkpoints")
    parser.add_argument('--config', required=True, help='Path to experiment configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Process the data
    summary_all, summary_correct, summary_incorrect, class_correct_dfs, class_incorrect_dfs, summary_dir = process_checkpoint_data(config)
    
    # Create visualizations
    create_visualizations(summary_all, summary_correct, summary_incorrect, 
                         class_correct_dfs, class_incorrect_dfs, summary_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()