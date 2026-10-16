import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

def analyze_gi_vs_singular_values(config_path):
    """Analyze GI vs singular values for all classes and checkpoints."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the root experiment directory and output directory
    root_dir = os.path.expanduser(config['output_dir'])
    output_dir = os.path.expanduser(config['gi_analysis']['analysis_output_dir'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith('class_')])
    
    # Get list of all checkpoint directories
    checkpoint_names = set()
    for class_dir in class_dirs:
        class_path = os.path.join(root_dir, class_dir)
        for correctness in ['correct', 'incorrect']:
            correctness_path = os.path.join(class_path, correctness)
            if not os.path.exists(correctness_path):
                continue
            
            for sample_dir in os.listdir(correctness_path):
                if not sample_dir.startswith('sample_'):
                    continue
                
                sample_path = os.path.join(correctness_path, sample_dir)
                for epoch_dir in os.listdir(sample_path):
                    if epoch_dir.startswith('model_epoch'):
                        checkpoint_names.add(epoch_dir)
    
    checkpoint_names = sorted(list(checkpoint_names))
    print(f"Found {len(checkpoint_names)} checkpoints to analyze")
    
    # Process each checkpoint
    for checkpoint_name in checkpoint_names:
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        
        # Create output directory for this checkpoint
        checkpoint_output_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        
        # Data structures to store results
        correct_gi_by_class = defaultdict(list)
        incorrect_gi_by_class = defaultdict(list)
        correct_sv_by_class = defaultdict(list)
        incorrect_sv_by_class = defaultdict(list)
        
        # Collect data for each class
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            class_num = int(class_dir.split('_')[1])
            class_path = os.path.join(root_dir, class_dir)
            
            # Process correctly classified samples
            correct_data = process_class_samples(
                class_path, 'correct', checkpoint_name, class_num
            )
            if correct_data:
                correct_gi_by_class[class_num].extend(correct_data['gi_data'])
                correct_sv_by_class[class_num].extend(correct_data['sv_data'])
            
            # Process incorrectly classified samples
            incorrect_data = process_class_samples(
                class_path, 'incorrect', checkpoint_name, class_num
            )
            if incorrect_data:
                incorrect_gi_by_class[class_num].extend(incorrect_data['gi_data'])
                incorrect_sv_by_class[class_num].extend(incorrect_data['sv_data'])
        
        # Create dataframes from collected data
        correct_df = create_dataframe_from_data(correct_gi_by_class, correct_sv_by_class)
        incorrect_df = create_dataframe_from_data(incorrect_gi_by_class, incorrect_sv_by_class)
        
        # Save dataframes to CSV
        correct_csv_path = os.path.join(checkpoint_output_dir, f"{checkpoint_name}_correct.csv")
        incorrect_csv_path = os.path.join(checkpoint_output_dir, f"{checkpoint_name}_incorrect.csv")
        
        correct_df.to_csv(correct_csv_path, index=False)
        incorrect_df.to_csv(incorrect_csv_path, index=False)
        
        print(f"Saved CSV files: {correct_csv_path}, {incorrect_csv_path}")
        
        # Generate plots
        generate_plots(
            correct_df, incorrect_df, checkpoint_output_dir, checkpoint_name
        )
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

def process_class_samples(class_path, correctness, checkpoint_name, class_num):
    """Process samples for a specific class and correctness."""
    correctness_path = os.path.join(class_path, correctness)
    if not os.path.exists(correctness_path):
        return None
    
    gi_data = []
    sv_data = []
    
    for sample_dir in os.listdir(correctness_path):
        if not sample_dir.startswith('sample_'):
            continue
        
        sample_path = os.path.join(correctness_path, sample_dir)
        checkpoint_path = os.path.join(sample_path, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            continue
        
        # Check if sample is actually correctly classified for this checkpoint
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # For each checkpoint, determine if the sample is correctly classified
        is_correct = metadata['predicted_class'] == metadata['true_class']
        
        # If we're processing 'correct' but the sample is incorrect for this checkpoint, skip
        # If we're processing 'incorrect' but the sample is correct for this checkpoint, skip
        if (correctness == 'correct' and not is_correct) or (correctness == 'incorrect' and is_correct):
            continue
        
        # Load GI results
        gi_results_path = os.path.join(checkpoint_path, "gi_results.npz")
        if not os.path.exists(gi_results_path):
            continue
        
        gi_results = np.load(gi_results_path)
        total_gi = gi_results['total_gi']
        positive_gi = gi_results['positive_gi']
        negative_gi = gi_results['negative_gi']
        
        # Load singular values
        svd_data_path = os.path.join(checkpoint_path, "svd_data.npz")
        if os.path.exists(svd_data_path):
            svd_data = np.load(svd_data_path)
            singular_values = svd_data['S']
        else:
            # Fallback to old format
            sv_path = os.path.join(checkpoint_path, "svd_S.npy")
            if not os.path.exists(sv_path):
                print(f"Warning: Singular values not found for {checkpoint_path}")
                continue
            singular_values = np.load(sv_path)
        
        # Make sure both arrays are the same length
        # If singular values are shorter, pad with zeros
        max_length = max(len(singular_values), len(total_gi))
        
        if len(singular_values) < max_length:
            padded_sv = np.zeros(max_length)
            padded_sv[:len(singular_values)] = singular_values
            singular_values = padded_sv
        
        if len(total_gi) < max_length:
            padded_total_gi = np.zeros(max_length)
            padded_total_gi[:len(total_gi)] = total_gi
            total_gi = padded_total_gi
            
            padded_positive_gi = np.zeros(max_length)
            padded_positive_gi[:len(positive_gi)] = positive_gi
            positive_gi = padded_positive_gi
            
            padded_negative_gi = np.zeros(max_length)
            padded_negative_gi[:len(negative_gi)] = negative_gi
            negative_gi = padded_negative_gi
        
        # Add data to lists
        for rsv_idx in range(max_length):
            gi_data.append({
                'class': class_num,
                'sample_idx': int(sample_dir.split('_')[1]),
                'rsv_idx': rsv_idx,
                'gi_length': total_gi[rsv_idx],
                'positive_gi': positive_gi[rsv_idx],
                'negative_gi': negative_gi[rsv_idx],
                'log_gi': np.log(total_gi[rsv_idx] + 1e-10),  # Add small epsilon to avoid log(0)
            })
            
            sv_data.append({
                'class': class_num,
                'sample_idx': int(sample_dir.split('_')[1]),
                'rsv_idx': rsv_idx,
                'singular_value': singular_values[rsv_idx] if rsv_idx < len(singular_values) else 0,
                'log_sv': np.log(singular_values[rsv_idx] + 1e-10) if rsv_idx < len(singular_values) else np.log(1e-10),
            })
    
    return {
        'gi_data': gi_data,
        'sv_data': sv_data
    }

def create_dataframe_from_data(gi_by_class, sv_by_class):
    """Create a pandas DataFrame from the collected data."""
    all_data = []
    
    for class_num in gi_by_class.keys():
        for i in range(len(gi_by_class[class_num])):
            gi_entry = gi_by_class[class_num][i]
            sv_entry = sv_by_class[class_num][i]
            
            # Ensure we're matching the correct entries
            assert gi_entry['class'] == sv_entry['class']
            assert gi_entry['sample_idx'] == sv_entry['sample_idx']
            assert gi_entry['rsv_idx'] == sv_entry['rsv_idx']
            
            combined_entry = {
                'class': class_num,
                'sample_idx': gi_entry['sample_idx'],
                'rsv_idx': gi_entry['rsv_idx'],
                'gi_length': gi_entry['gi_length'],
                'positive_gi': gi_entry['positive_gi'],
                'negative_gi': gi_entry['negative_gi'],
                'log_gi': gi_entry['log_gi'],
                'singular_value': sv_entry['singular_value'],
                'log_sv': sv_entry['log_sv']
            }
            
            all_data.append(combined_entry)
    
    return pd.DataFrame(all_data)

    
def compute_gi_volume(df):
    """
    Compute GI volume for each sample by summing log(GI) values along all RSV directions.
    Returns a DataFrame with class, sample_idx, and volume.
    """
    # Group by class and sample_idx
    volume_data = []
    
    # Get unique combination of class and sample
    samples = df[['class', 'sample_idx']].drop_duplicates().values
    
    for class_num, sample_idx in samples:
        # Get all GI values for this sample
        sample_data = df[(df['class'] == class_num) & (df['sample_idx'] == sample_idx)]
        
        # Calculate log(GI) and sum
        log_gis = np.log(sample_data['gi_length'] + 1e-10)  # Add small epsilon to avoid log(0)
        volume = np.sum(log_gis)
        
        volume_data.append({
            'class': class_num,
            'sample_idx': sample_idx,
            'gi_volume': volume,
            'num_rsv': len(sample_data)  # Store number of RSVs for reference
        })
    
    return pd.DataFrame(volume_data)

def generate_plots(correct_df, incorrect_df, output_dir, checkpoint_name):
    """Generate plots for GI vs singular values and GI vs RSV index."""
    print(f"Generating plots for {checkpoint_name}...")
    
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    if not correct_df.empty:
        correct_volumes = compute_gi_volume(correct_df)
        
        # Calculate average volume by class
        class_volumes = correct_volumes.groupby('class')['gi_volume'].agg(['mean', 'std']).reset_index()
        
        # Plot average volume by class
        plt.figure(figsize=(12, 6))
        plt.bar(class_volumes['class'], class_volumes['mean'], yerr=class_volumes['std'], 
               color='blue', alpha=0.7)
        plt.title(f'Average GI Volume by Class (Correctly Classified) - {checkpoint_name}')
        plt.xlabel('Class')
        plt.ylabel('GI Volume (sum of log(GI))')
        plt.xticks(range(10))
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'gi_volume_by_class_correct.png'), dpi=300)
        plt.close()
        
        # Save volume data to CSV
        correct_volumes.to_csv(os.path.join(output_dir, f"{checkpoint_name}_correct_volumes.csv"), index=False)
    
    if not incorrect_df.empty:
        incorrect_volumes = compute_gi_volume(incorrect_df)
        
        # Calculate average volume by class
        class_volumes = incorrect_volumes.groupby('class')['gi_volume'].agg(['mean', 'std']).reset_index()
        
        # Plot average volume by class
        plt.figure(figsize=(12, 6))
        plt.bar(class_volumes['class'], class_volumes['mean'], yerr=class_volumes['std'], 
               color='red', alpha=0.7)
        plt.title(f'Average GI Volume by Class (Incorrectly Classified) - {checkpoint_name}')
        plt.xlabel('Class')
        plt.ylabel('GI Volume (sum of log(GI))')
        plt.xticks(range(10))
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'gi_volume_by_class_incorrect.png'), dpi=300)
        plt.close()
        
        # Save volume data to CSV
        incorrect_volumes.to_csv(os.path.join(output_dir, f"{checkpoint_name}_incorrect_volumes.csv"), index=False)
    
    # If both datasets exist, create a comparison plot
    if not correct_df.empty and not incorrect_df.empty:
        correct_volumes = compute_gi_volume(correct_df)
        incorrect_volumes = compute_gi_volume(incorrect_df)
        
        correct_class_volumes = correct_volumes.groupby('class')['gi_volume'].mean().reset_index()
        incorrect_class_volumes = incorrect_volumes.groupby('class')['gi_volume'].mean().reset_index()
        
        # Merge the data
        merged_volumes = pd.merge(correct_class_volumes, incorrect_class_volumes, 
                                 on='class', suffixes=('_correct', '_incorrect'))
        
        # Create a DataFrame suitable for plotting
        plot_data = pd.DataFrame({
            'Class': np.concatenate([merged_volumes['class'], merged_volumes['class']]),
            'Volume': np.concatenate([merged_volumes['gi_volume_correct'], merged_volumes['gi_volume_incorrect']]),
            'Type': ['Correct']*len(merged_volumes) + ['Incorrect']*len(merged_volumes)
        })
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Class', y='Volume', hue='Type', data=plot_data)
        plt.title(f'Average GI Volume by Class Comparison - {checkpoint_name}')
        plt.xlabel('Class')
        plt.ylabel('GI Volume (sum of log(GI))')
        plt.legend(title='Classification')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f'gi_volume_comparison.png'), dpi=300)
        plt.close()

    # 1. Average GI vs RSV index across all classes (correctly classified)
    if not correct_df.empty:
        plt.figure(figsize=(10, 6))
        gi_by_rsv = correct_df.groupby('rsv_idx')['gi_length'].agg(['mean', 'std']).reset_index()
        
        plt.plot(gi_by_rsv['rsv_idx'], gi_by_rsv['mean'], color='blue')
        plt.fill_between(
            gi_by_rsv['rsv_idx'],
            gi_by_rsv['mean'] - gi_by_rsv['std'],
            gi_by_rsv['mean'] + gi_by_rsv['std'],
            alpha=0.3,
            color='blue'
        )
        plt.title(f'Average GI vs RSV Index (Correctly Classified) - {checkpoint_name}')
        plt.xlabel('RSV Index')
        plt.ylabel('GI Length')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'avg_gi_vs_rsv_correct.png'), dpi=300)
        plt.close()
        
        # 2. Demonstrate inverse proportionality: GI * Singular Value ~ Constant
        # Filter data to remove zeros and very small values
        valid_data = correct_df[(correct_df['singular_value'] > 1e-6) & (correct_df['gi_length'] > 1e-6)]
        
        if not valid_data.empty:
            plt.figure(figsize=(10, 6))
            
            # Calculate product of GI * Singular Value
            valid_data['product'] = valid_data['gi_length'] * valid_data['singular_value']
            
            # Group by RSV index bins for better visualization
            valid_data['rsv_bin'] = pd.cut(valid_data['rsv_idx'], bins=20)
            product_by_bin = valid_data.groupby('rsv_bin')['product'].mean().reset_index()
            product_by_bin['bin_center'] = product_by_bin['rsv_bin'].apply(lambda x: (x.left + x.right) / 2)
            
            plt.scatter(product_by_bin['bin_center'], product_by_bin['product'], s=50, color='blue')
            plt.axhline(y=valid_data['product'].mean(), color='r', linestyle='--', 
                       label=f'Mean product: {valid_data["product"].mean():.4f}')
            
            plt.title(f'GI × Singular Value (Correctly Classified) - {checkpoint_name}')
            plt.xlabel('RSV Index')
            plt.ylabel('GI Length × Singular Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'gi_times_sv_correct.png'), dpi=300)
            plt.close()
            
            # 3. Direct plot of inverse proportionality
            plt.figure(figsize=(10, 6))
            
            # Group data by singular value bins
            valid_data['sv_bin'] = pd.cut(valid_data['singular_value'], bins=np.logspace(
                np.log10(valid_data['singular_value'].min()),
                np.log10(valid_data['singular_value'].max()),
                20))
            
            gi_by_sv = valid_data.groupby('sv_bin').agg({
                'singular_value': 'mean',
                'gi_length': 'mean'
            }).reset_index()
            
            # Plot GI vs SV
            plt.scatter(gi_by_sv['singular_value'], gi_by_sv['gi_length'], s=50, color='blue')
            
            # Add power law fit (y = a*x^b where b should be close to -1 for inverse proportionality)
            from scipy.optimize import curve_fit
            
            def power_law(x, a, b):
                return a * np.power(x, b)
            
            try:
                popt, pcov = curve_fit(power_law, gi_by_sv['singular_value'], gi_by_sv['gi_length'])
                x_smooth = np.logspace(np.log10(gi_by_sv['singular_value'].min()), 
                                      np.log10(gi_by_sv['singular_value'].max()), 100)
                plt.plot(x_smooth, power_law(x_smooth, *popt), 'r-', 
                        label=f'Power law: y = {popt[0]:.2f} × x^({popt[1]:.2f})')
                
                # Add reference line for perfect inverse proportionality (y = c/x)
                c = np.mean(valid_data['product'])
                plt.plot(x_smooth, c/x_smooth, 'g--', 
                        label=f'Perfect inverse: y = {c:.2f}/x')
            except:
                print("Could not fit power law to correctly classified data")
            
            plt.title(f'GI vs Singular Value (Correctly Classified) - {checkpoint_name}')
            plt.xlabel('Singular Value')
            plt.ylabel('GI Length')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'gi_vs_sv_inverse_correct.png'), dpi=300)
            plt.close()
    
    # 4. Average GI vs RSV index across all classes (incorrectly classified)
    if not incorrect_df.empty:
        plt.figure(figsize=(10, 6))
        gi_by_rsv = incorrect_df.groupby('rsv_idx')['gi_length'].agg(['mean', 'std']).reset_index()
        
        plt.plot(gi_by_rsv['rsv_idx'], gi_by_rsv['mean'], color='red')
        plt.fill_between(
            gi_by_rsv['rsv_idx'],
            gi_by_rsv['mean'] - gi_by_rsv['std'],
            gi_by_rsv['mean'] + gi_by_rsv['std'],
            alpha=0.3,
            color='red'
        )
        plt.title(f'Average GI vs RSV Index (Incorrectly Classified) - {checkpoint_name}')
        plt.xlabel('RSV Index')
        plt.ylabel('GI Length')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'avg_gi_vs_rsv_incorrect.png'), dpi=300)
        plt.close()
        
        # 5. Demonstrate inverse proportionality for incorrect samples
        valid_data = incorrect_df[(incorrect_df['singular_value'] > 1e-6) & (incorrect_df['gi_length'] > 1e-6)]
        
        if not valid_data.empty:
            plt.figure(figsize=(10, 6))
            
            # Calculate product of GI * Singular Value
            valid_data['product'] = valid_data['gi_length'] * valid_data['singular_value']
            
            # Group by RSV index bins for better visualization
            valid_data['rsv_bin'] = pd.cut(valid_data['rsv_idx'], bins=20)
            product_by_bin = valid_data.groupby('rsv_bin')['product'].mean().reset_index()
            product_by_bin['bin_center'] = product_by_bin['rsv_bin'].apply(lambda x: (x.left + x.right) / 2)
            
            plt.scatter(product_by_bin['bin_center'], product_by_bin['product'], s=50, color='red')
            plt.axhline(y=valid_data['product'].mean(), color='r', linestyle='--', 
                       label=f'Mean product: {valid_data["product"].mean():.4f}')
            
            plt.title(f'GI × Singular Value (Incorrectly Classified) - {checkpoint_name}')
            plt.xlabel('RSV Index')
            plt.ylabel('GI Length × Singular Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'gi_times_sv_incorrect.png'), dpi=300)
            plt.close()
            
            # 6. Direct plot of inverse proportionality for incorrect samples
            plt.figure(figsize=(10, 6))
            
            # Group data by singular value bins
            valid_data['sv_bin'] = pd.cut(valid_data['singular_value'], bins=np.logspace(
                np.log10(valid_data['singular_value'].min()),
                np.log10(valid_data['singular_value'].max()),
                20))
            
            gi_by_sv = valid_data.groupby('sv_bin').agg({
                'singular_value': 'mean',
                'gi_length': 'mean'
            }).reset_index()
            
            # Plot GI vs SV
            plt.scatter(gi_by_sv['singular_value'], gi_by_sv['gi_length'], s=50, color='red')
            
            # Add power law fit
            try:
                popt, pcov = curve_fit(power_law, gi_by_sv['singular_value'], gi_by_sv['gi_length'])
                x_smooth = np.logspace(np.log10(gi_by_sv['singular_value'].min()), 
                                      np.log10(gi_by_sv['singular_value'].max()), 100)
                plt.plot(x_smooth, power_law(x_smooth, *popt), 'r-', 
                        label=f'Power law: y = {popt[0]:.2f} × x^({popt[1]:.2f})')
                
                # Add reference line for perfect inverse proportionality
                c = np.mean(valid_data['product'])
                plt.plot(x_smooth, c/x_smooth, 'g--', 
                        label=f'Perfect inverse: y = {c:.2f}/x')
            except:
                print("Could not fit power law to incorrectly classified data")
            
            plt.title(f'GI vs Singular Value (Incorrectly Classified) - {checkpoint_name}')
            plt.xlabel('Singular Value')
            plt.ylabel('GI Length')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'gi_vs_sv_inverse_incorrect.png'), dpi=300)
            plt.close()
    
    # 7. Comparison plot (if both datasets exist)
    if not correct_df.empty and not incorrect_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Get mean GI by RSV for both datasets
        correct_mean = correct_df.groupby('rsv_idx')['gi_length'].mean()
        incorrect_mean = incorrect_df.groupby('rsv_idx')['gi_length'].mean()
        
        # Find common RSV indices
        common_indices = sorted(set(correct_mean.index).intersection(set(incorrect_mean.index)))
        
        plt.plot(common_indices, correct_mean.loc[common_indices], label='Correctly Classified', color='blue')
        plt.plot(common_indices, incorrect_mean.loc[common_indices], label='Incorrectly Classified', color='red')
        
        plt.title(f'GI vs RSV Index Comparison - {checkpoint_name}')
        plt.xlabel('RSV Index')
        plt.ylabel('Mean GI Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'gi_vs_rsv_comparison.png'), dpi=300)
        plt.close()
        
        # 8. Compare inverse proportionality for correct vs incorrect
        correct_valid = correct_df[(correct_df['singular_value'] > 1e-6) & (correct_df['gi_length'] > 1e-6)]
        incorrect_valid = incorrect_df[(incorrect_df['singular_value'] > 1e-6) & (incorrect_df['gi_length'] > 1e-6)]
        
        if not correct_valid.empty and not incorrect_valid.empty:
            # Bin both datasets
            sv_min = min(correct_valid['singular_value'].min(), incorrect_valid['singular_value'].min())
            sv_max = max(correct_valid['singular_value'].max(), incorrect_valid['singular_value'].max())
            
            bins = np.logspace(np.log10(sv_min), np.log10(sv_max), 20)
            
            correct_valid['sv_bin'] = pd.cut(correct_valid['singular_value'], bins)
            incorrect_valid['sv_bin'] = pd.cut(incorrect_valid['singular_value'], bins)
            
            correct_binned = correct_valid.groupby('sv_bin').agg({
                'singular_value': 'mean',
                'gi_length': 'mean'
            }).reset_index()
            
            incorrect_binned = incorrect_valid.groupby('sv_bin').agg({
                'singular_value': 'mean',
                'gi_length': 'mean'
            }).reset_index()
            
            plt.figure(figsize=(10, 6))
            
            # Plot both datasets
            plt.scatter(correct_binned['singular_value'], correct_binned['gi_length'], 
                       s=50, color='blue', label='Correctly Classified')
            plt.scatter(incorrect_binned['singular_value'], incorrect_binned['gi_length'], 
                       s=50, color='red', label='Incorrectly Classified')
            
            # Add power law fits
            try:
                # Fit for correct samples
                popt_correct, _ = curve_fit(power_law, correct_binned['singular_value'], correct_binned['gi_length'])
                
                # Fit for incorrect samples
                popt_incorrect, _ = curve_fit(power_law, incorrect_binned['singular_value'], incorrect_binned['gi_length'])
                
                # Create smooth x values for plotting
                x_smooth = np.logspace(np.log10(sv_min), np.log10(sv_max), 100)
                
                # Plot fitted curves
                plt.plot(x_smooth, power_law(x_smooth, *popt_correct), '--', color='blue',
                        label=f'Correct fit: y = {popt_correct[0]:.2f} × x^({popt_correct[1]:.2f})')
                
                plt.plot(x_smooth, power_law(x_smooth, *popt_incorrect), '--', color='red',
                        label=f'Incorrect fit: y = {popt_incorrect[0]:.2f} × x^({popt_incorrect[1]:.2f})')
                
                # Calculate average products
                c_correct = np.mean(correct_valid['gi_length'] * correct_valid['singular_value'])
                c_incorrect = np.mean(incorrect_valid['gi_length'] * incorrect_valid['singular_value'])
                
                # Plot perfect inverse proportionality reference lines
                plt.plot(x_smooth, c_correct/x_smooth, ':', color='blue', alpha=0.7,
                        label=f'Perfect inverse (correct): y = {c_correct:.2f}/x')
                
                plt.plot(x_smooth, c_incorrect/x_smooth, ':', color='red', alpha=0.7,
                        label=f'Perfect inverse (incorrect): y = {c_incorrect:.2f}/x')
                
            except Exception as e:
                print(f"Could not fit power law to comparison data: {e}")
            
            plt.title(f'GI vs Singular Value Comparison - {checkpoint_name}')
            plt.xlabel('Singular Value')
            plt.ylabel('GI Length')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'gi_vs_sv_comparison.png'), dpi=300)
            plt.close()

def analyze_all_checkpoints(config_path):
    """Analyze GI vs singular values across all checkpoints."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the output directory
    output_dir = os.path.expanduser(config['gi_analysis']['analysis_output_dir'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a list of all checkpoint folders in the output directory
    checkpoint_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('model_epoch')]
    checkpoint_dirs = sorted(checkpoint_dirs)
    
    if not checkpoint_dirs:
        print("No checkpoint data found. Run the analysis first.")
        return
    
    print(f"Found data for {len(checkpoint_dirs)} checkpoints")
    
    # Create dataframes to store the aggregated data for each checkpoint
    correct_data_by_checkpoint = {}
    incorrect_data_by_checkpoint = {}
    
    # Load data for each checkpoint
    for checkpoint in checkpoint_dirs:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        
        # Load correct samples data
        correct_csv = os.path.join(checkpoint_path, f"{checkpoint}_correct.csv")
        if os.path.exists(correct_csv):
            correct_df = pd.read_csv(correct_csv)
            
            # Group by singular value bins for better visualization
            correct_df = correct_df[correct_df['singular_value'] > 1e-9]  # Filter out zero or very small values
            if not correct_df.empty:
                # Create log-spaced bins for singular values
                bins = np.logspace(
                    np.log10(correct_df['singular_value'].min()), 
                    np.log10(correct_df['singular_value'].max()),
                    20
                )
                
                correct_df['sv_bin'] = pd.cut(correct_df['singular_value'], bins)
                
                # Calculate mean GI and SV for each bin
                binned_data = correct_df.groupby('sv_bin').agg({
                    'singular_value': 'mean',
                    'gi_length': 'mean',
                    'log_gi': 'mean'
                }).reset_index()
                
                correct_data_by_checkpoint[checkpoint] = binned_data
        
        # Load incorrect samples data
        incorrect_csv = os.path.join(checkpoint_path, f"{checkpoint}_incorrect.csv")
        if os.path.exists(incorrect_csv):
            incorrect_df = pd.read_csv(incorrect_csv)
            
            # Group by singular value bins for better visualization
            incorrect_df = incorrect_df[incorrect_df['singular_value'] > 1e-9]  # Filter out zero or very small values
            if not incorrect_df.empty:
                # Create log-spaced bins for singular values
                bins = np.logspace(
                    np.log10(incorrect_df['singular_value'].min()), 
                    np.log10(incorrect_df['singular_value'].max()),
                    20
                )
                
                incorrect_df['sv_bin'] = pd.cut(incorrect_df['singular_value'], bins)
                
                # Calculate mean GI and SV for each bin
                binned_data = incorrect_df.groupby('sv_bin').agg({
                    'singular_value': 'mean',
                    'gi_length': 'mean',
                    'log_gi': 'mean'
                }).reset_index()
                
                incorrect_data_by_checkpoint[checkpoint] = binned_data
    
    # Plot GI vs singular value for all checkpoints (correctly classified)
    if correct_data_by_checkpoint:
        plot_checkpoint_comparison(correct_data_by_checkpoint, output_dir, 'correct')
    
    # Plot GI vs singular value for all checkpoints (incorrectly classified)
    if incorrect_data_by_checkpoint:
        plot_checkpoint_comparison(incorrect_data_by_checkpoint, output_dir, 'incorrect')
    
    print(f"All checkpoint comparison plots saved to {output_dir}")

def plot_checkpoint_comparison(data_by_checkpoint, output_dir, classification_type):
    """Create comparison plots across all checkpoints."""
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create a color map for the checkpoints
    checkpoints = sorted(list(data_by_checkpoint.keys()))
    num_checkpoints = len(checkpoints)
    colors = plt.cm.viridis(np.linspace(0, 1, num_checkpoints))
    
    # 1. Plot GI vs singular value for all checkpoints (log-log scale)
    plt.figure(figsize=(12, 8))
    
    for i, checkpoint in enumerate(checkpoints):
        binned_data = data_by_checkpoint[checkpoint]
        plt.scatter(binned_data['singular_value'], binned_data['gi_length'], 
                   color=colors[i], s=30, alpha=0.7)
        
        # Add power law fit
        from scipy.optimize import curve_fit
        
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            popt, _ = curve_fit(power_law, binned_data['singular_value'], binned_data['gi_length'])
            
            # Create smooth x values for plotting
            x_smooth = np.logspace(
                np.log10(binned_data['singular_value'].min()),
                np.log10(binned_data['singular_value'].max()),
                100
            )
            
            plt.plot(x_smooth, power_law(x_smooth, *popt), '-', color=colors[i],
                    label=f"{checkpoint}: y = {popt[0]:.2f}x^({popt[1]:.2f})")
            
        except Exception as e:
            print(f"Could not fit power law for {checkpoint}: {e}")
            plt.plot(binned_data['singular_value'], binned_data['gi_length'], '-', color=colors[i],
                    label=checkpoint)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'GI vs Singular Value Across Checkpoints ({classification_type.title()})')
    plt.xlabel('Singular Value')
    plt.ylabel('GI Length')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gi_vs_sv_all_checkpoints_{classification_type}.png'), dpi=300)
    plt.close()
    
    # 2. Plot log(GI) vs log(singular value) for all checkpoints (linear scale)
    plt.figure(figsize=(12, 8))
    
    for i, checkpoint in enumerate(checkpoints):
        binned_data = data_by_checkpoint[checkpoint]
        log_sv = np.log(binned_data['singular_value'])
        log_gi = np.log(binned_data['gi_length'])
        
        plt.scatter(log_sv, log_gi, color=colors[i], s=30, alpha=0.7)
        
        # Add linear fit
        slope, intercept, r_value, _, _ = stats.linregress(log_sv, log_gi)
        
        plt.plot(log_sv, intercept + slope*log_sv, '-', color=colors[i],
                label=f"{checkpoint}: slope = {slope:.2f}, r² = {r_value**2:.2f}")
    
    plt.title(f'Log(GI) vs Log(Singular Value) Across Checkpoints ({classification_type.title()})')
    plt.xlabel('Log(Singular Value)')
    plt.ylabel('Log(GI Length)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'log_gi_vs_log_sv_all_checkpoints_{classification_type}.png'), dpi=300)
    plt.close()
    
    # 3. Plot power law exponents across checkpoints
    exponents = []
    for checkpoint in checkpoints:
        binned_data = data_by_checkpoint[checkpoint]
        log_sv = np.log(binned_data['singular_value'])
        log_gi = np.log(binned_data['gi_length'])
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(log_sv, log_gi)
            exponents.append({
                'checkpoint': checkpoint,
                'exponent': slope,
                'r_squared': r_value**2
            })
        except:
            continue
    
    if exponents:
        exponent_df = pd.DataFrame(exponents)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(exponent_df)), exponent_df['exponent'], 'o-', color='blue')
        
        # Add reference line at y = -1 (perfect inverse proportionality)
        plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7, 
                   label='Perfect inverse proportionality (exponent = -1)')
        
        plt.title(f'Power Law Exponent Across Checkpoints ({classification_type.title()})')
        plt.xlabel('Checkpoint Index')
        plt.ylabel('Exponent (slope of log-log plot)')
        plt.xticks(range(len(exponent_df)), exponent_df['checkpoint'], rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'power_law_exponent_trend_{classification_type}.png'), dpi=300)
        plt.close()
        
        # Save exponents to CSV
        exponent_df.to_csv(os.path.join(output_dir, f'power_law_exponents_{classification_type}.csv'), index=False)

# Add the below code at the end of the main function after calling analyze_gi_vs_singular_values
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GI vs singular values")
    parser.add_argument('--config', required=True, help='Path to experiment configuration file')
    
    args = parser.parse_args()
    analyze_gi_vs_singular_values(args.config)
    
    # After analyzing individual checkpoints, create comparison plots across all checkpoints
    analyze_all_checkpoints(args.config)
