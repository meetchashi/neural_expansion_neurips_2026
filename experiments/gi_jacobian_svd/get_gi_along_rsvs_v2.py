import os
import numpy as np
import torch
import json
import yaml
from pathlib import Path
import sys
import importlib.util
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from src.analysis.gi_calculation import compute_GIs_for_sample, compute_GI


def compute_gi_along_rsvs_batched_optimized(model, sample, V, true_class, max_alpha=3.0, step_size=0.01, rsv_scale=56.0, batch_size=100, device='cuda'):
    """
    Compute GIs along RSV directions using the optimized batch function
    while ensuring identical behavior to the original approach
    """
    model.eval()
    model = model.to(device)
    
    # Ensure sample has correct dimensions [C, H, W]
    if sample.dim() == 4 and sample.shape[0] == 1:
        sample = sample.squeeze(0)
    
    # Move sample to device
    sample = sample.to(device)
    
    # Get original shape for reshaping RSVs
    if sample.dim() == 3:
        C, H, W = sample.shape
    else:
        H, W = sample.shape[-2:]
        C = 1 if sample.dim() == 2 else sample.shape[0]
    
    original_shape = (C, H, W)
    
    # Verify original prediction
    with torch.no_grad():
        outputs = model(sample.unsqueeze(0))
        pred_class = torch.argmax(outputs, dim=1).item()
    
    if pred_class != true_class:
        print(f"Warning: Original prediction {pred_class} doesn't match true class {true_class}")
        return np.zeros(V.shape[1]), np.zeros(V.shape[1]), np.zeros(V.shape[1])
    
    # Process RSVs in batches
    all_pos_gi = []
    all_neg_gi = []
    num_rsvs = V.shape[1]
    
    for batch_start in range(0, num_rsvs, batch_size):
        batch_end = min(batch_start + batch_size, num_rsvs)
        
        # Extract and prepare batch of RSVs
        batch_rsv_tensors = []
        for i in range(batch_start, batch_end):
            rsv = V[:, i]
            rsv_norm = np.linalg.norm(rsv)
            
            if rsv_norm > 0:
                # Scale the RSV
                rsv = rsv / rsv_norm * rsv_scale
                
                # Reshape to match input dimensions
                rsv_reshaped = rsv.reshape(original_shape)
                rsv_tensor = torch.tensor(rsv_reshaped, dtype=torch.float32)
                batch_rsv_tensors.append(rsv_tensor)
            else:
                # Add a zero tensor for consistency
                batch_rsv_tensors.append(torch.zeros(original_shape, dtype=torch.float32))
        
        # Stack the tensors to create a batch
        batch_rsv_tensor = torch.stack(batch_rsv_tensors).to(device)
        
        # Create a batch of sample tensors for the reference point
        sample_tensor = sample.unsqueeze(0).expand(len(batch_rsv_tensor), -1, -1, -1)
        
        # Perturbed samples (sample + RSV direction)
        perturbed_samples = sample_tensor + batch_rsv_tensor
        
        # Calculate GIs using the compute_GI function
        pos_batch, neg_batch = compute_GI(
            model=model,
            x1=sample_tensor,
            x2=perturbed_samples,
            direction_normalized=False,  # Already normalized above
            correct_class=true_class,
            step_size=step_size,
            max_eps=max_alpha,
            device=device
        )
        
        all_pos_gi.extend(pos_batch)
        all_neg_gi.extend(neg_batch)
    
    # Process the results
    positive_gi = np.array(all_pos_gi)
    negative_gi = np.array(all_neg_gi)
    total_gi = positive_gi + negative_gi
    
    return positive_gi, negative_gi, total_gi

def load_model_from_checkpoint(checkpoint_path, model_config_path):
    """Load model from checkpoint."""
    # Load the model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Get model architecture details
    model_name = model_config['model']['name']
    model_args = model_config['model'].get('args', {})
    
    # Import the module containing the model
    module_path = model_config['model'].get('module', 'src.models')
    model_file = model_config['model'].get('file', model_name.lower())
    
    # Add the project root to system path if needed
    project_root = Path(model_config_path).parents[2]  # Assuming config is in configs/training/
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Import the module
    module = importlib.import_module(f"{module_path}.{model_file}")
    
    # Get the model class and instantiate it
    model_class = getattr(module, model_name)
    model = model_class(**model_args)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    return model

def process_experiments(config_path):
    """Process all samples in the experiment directory."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the root experiment directory
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    root_dir = os.path.expanduser(config['output_dir'])
    if not os.path.exists(root_dir):
        raise ValueError(f"Experiment directory does not exist: {root_dir}")
    
    # Get model config path
    model_config_path = os.path.expanduser(config['model']['config_path'])
    
    # Get GI analysis parameters
    gi_params = config.get('gi_analysis', {})
    max_alpha = gi_params.get('max_alpha', 3.0)
    step_size = gi_params.get('step_size', 0.01)
    device = gi_params.get('device', 'cuda')
    batch_size = gi_params.get('rsv_batch_size', 100)
    rsv_scale = gi_params.get('rsv_scale', 56.0)
    analysis_output_dir = os.path.expanduser(gi_params.get('analysis_output_dir', os.path.join(root_dir, 'rsv_gi_analysis')))
    
    # Create output directory if it doesn't exist
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GI Parameters - Max Alpha: {max_alpha}, Step Size: {step_size}")
    print(f"RSV Parameters - Scale: {rsv_scale}, Batch Size: {batch_size}")
    print(f"Output directory: {analysis_output_dir}")
    
    # Find all unique checkpoints
    checkpoint_names = set()
    for class_dir in os.listdir(root_dir):
        if not class_dir.startswith('class_'):
            continue
        
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
    print(f"Found {len(checkpoint_names)} unique checkpoints")
    
    # Process each checkpoint
    for checkpoint_name in checkpoint_names:
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        
        # Create checkpoint output directory
        checkpoint_output_dir = os.path.join(analysis_output_dir, checkpoint_name)
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        
        # Initialize aggregated results for the checkpoint
        all_correct_results = []
        all_incorrect_results = []
        
        # Process all samples for this checkpoint by class
        for class_dir in sorted(os.listdir(root_dir)):
            if not class_dir.startswith('class_'):
                continue
            
            class_path = os.path.join(root_dir, class_dir)
            class_num = int(class_dir.split('_')[1])
            print(f"\nProcessing class {class_num}...")
            
            # Initialize class-specific results
            class_correct_results = []
            class_incorrect_results = []
            
            for correctness in ['correct', 'incorrect']:
                results_list = class_correct_results if correctness == 'correct' else class_incorrect_results
                all_results_list = all_correct_results if correctness == 'correct' else all_incorrect_results
                
                correctness_path = os.path.join(class_path, correctness)
                if not os.path.exists(correctness_path):
                    continue
                
                for sample_dir in tqdm(sorted(os.listdir(correctness_path)), desc=f"Class {class_num} {correctness}"):
                    if not sample_dir.startswith('sample_'):
                        continue
                    
                    sample_path = os.path.join(correctness_path, sample_dir)
                    epoch_path = os.path.join(sample_path, checkpoint_name)
                    
                    # Skip if checkpoint doesn't exist for this sample
                    if not os.path.exists(epoch_path):
                        continue
                    
                    # Load metadata
                    metadata_path = os.path.join(epoch_path, "metadata.json")
                    if not os.path.exists(metadata_path):
                        print(f"Warning: Metadata not found: {metadata_path}")
                        continue
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract sample information
                    sample_idx = int(sample_dir.split('_')[1])
                    true_class = metadata['true_class']
                    predicted_class = metadata['predicted_class']
                    is_correct = metadata['is_correct']
                    
                    # Check if SVD data exists
                    svd_data_path = os.path.join(epoch_path, "svd_data.npz")
                    if not os.path.exists(svd_data_path):
                        print(f"Error: SVD data not found for {epoch_path}")
                        continue
                    
                    # Check if sample data exists
                    sample_data_path = os.path.join(sample_path, f"{sample_dir}_data.npy")
                    if not os.path.exists(sample_data_path):
                        print(f"Error: Sample data not found: {sample_data_path}")
                        continue
                    
                    # Load sample data
                    sample_data = np.load(sample_data_path)
                    x = torch.tensor(sample_data, dtype=torch.float32)
                    
                    # Load V matrix (right singular vectors)
                    svd_data = np.load(svd_data_path)
                    V = svd_data['V']
                    
                    # Load the model for this checkpoint
                    checkpoint_path = metadata['checkpoint']
                    try:
                        model = load_model_from_checkpoint(checkpoint_path, model_config_path)
                    except Exception as e:
                        print(f"Error loading model from {checkpoint_path}: {e}")
                        continue
                    
                    # Compute GI for each RSV direction using the correct function
                    try:
                        positive_gi, negative_gi, total_gi = compute_gi_along_rsvs_batched_optimized(
                            model=model,
                            sample=x,
                            V=V,
                            true_class=true_class,
                            max_alpha=max_alpha,
                            step_size=step_size,
                            rsv_scale=rsv_scale,
                            batch_size=batch_size,
                            device=device
                        )
                        
                        # Save results to original directory for future reference
                        gi_results_path = os.path.join(epoch_path, "gi_results.npz")
                        np.savez_compressed(
                            gi_results_path,
                            positive_gi=positive_gi,
                            negative_gi=negative_gi,
                            total_gi=total_gi
                        )
                        
                    except Exception as e:
                        print(f"Error computing GI for {epoch_path}: {e}")
                        continue
                    
                    # Load singular values
                    try:
                        singular_values = svd_data['S']  # Get directly from the svd_data we already loaded
                    except Exception as e:
                        print(f"Error loading singular values from {svd_data_path}: {e}")
                        singular_values = np.zeros_like(total_gi)
                    
                    # Calculate statistics - separate for positive and negative GIs
                    num_pos_gi_above_1 = np.sum(positive_gi > 1.0)
                    percent_pos_gi_above_1 = 100 * num_pos_gi_above_1 / len(positive_gi) if len(positive_gi) > 0 else 0
                    
                    num_neg_gi_above_1 = np.sum(negative_gi > 1.0)
                    percent_neg_gi_above_1 = 100 * num_neg_gi_above_1 / len(negative_gi) if len(negative_gi) > 0 else 0
                    
                    num_total_gi_above_1 = np.sum(total_gi > 1.0)
                    percent_total_gi_above_1 = 100 * num_total_gi_above_1 / len(total_gi) if len(total_gi) > 0 else 0
                    
                    # Create result dictionary
                    result = {
                        'sample_idx': sample_idx,
                        'class': class_num,
                        'true_class': true_class,
                        'predicted_class': predicted_class,
                        'is_correct': is_correct,
                        'num_rsv_directions': len(total_gi),
                        'percent_pos_gi_above_1': percent_pos_gi_above_1,
                        'percent_neg_gi_above_1': percent_neg_gi_above_1,
                        'percent_total_gi_above_1': percent_total_gi_above_1,
                        'min_pos_gi': np.min(positive_gi) if len(positive_gi) > 0 else 0,
                        'max_pos_gi': np.max(positive_gi) if len(positive_gi) > 0 else 0,
                        'mean_pos_gi': np.mean(positive_gi) if len(positive_gi) > 0 else 0,
                        'std_pos_gi': np.std(positive_gi) if len(positive_gi) > 0 else 0,
                        'median_pos_gi': np.median(positive_gi) if len(positive_gi) > 0 else 0,
                        'min_neg_gi': np.min(negative_gi) if len(negative_gi) > 0 else 0,
                        'max_neg_gi': np.max(negative_gi) if len(negative_gi) > 0 else 0,
                        'mean_neg_gi': np.mean(negative_gi) if len(negative_gi) > 0 else 0,
                        'std_neg_gi': np.std(negative_gi) if len(negative_gi) > 0 else 0,
                        'median_neg_gi': np.median(negative_gi) if len(negative_gi) > 0 else 0,
                        'min_total_gi': np.min(total_gi) if len(total_gi) > 0 else 0,
                        'max_total_gi': np.max(total_gi) if len(total_gi) > 0 else 0,
                        'mean_total_gi': np.mean(total_gi) if len(total_gi) > 0 else 0,
                        'std_total_gi': np.std(total_gi) if len(total_gi) > 0 else 0,
                        'median_total_gi': np.median(total_gi) if len(total_gi) > 0 else 0,
                    }
                    
                    # Add individual GIs and singular values
                    # Limit to specified number to avoid extremely wide CSVs
                    num_values_to_save = gi_params.get("max_rsvs_to_save", len(total_gi))
                    for i in range(min(num_values_to_save, len(total_gi))):
                        result[f'pos_gi_{i}'] = positive_gi[i]
                        result[f'neg_gi_{i}'] = negative_gi[i]
                        result[f'total_gi_{i}'] = total_gi[i]
                        result[f'singular_value_{i}'] = singular_values[i] if i < len(singular_values) else 0
                    
                    # Add to both class-specific and aggregated results lists
                    results_list.append(result)
                    all_results_list.append(result)
            
            # Save class-specific results
            if class_correct_results:
                class_correct_df = pd.DataFrame(class_correct_results)
                class_correct_csv = os.path.join(checkpoint_output_dir, f'class_{class_num}_correct_rsv_gi.csv')
                class_correct_df.to_csv(class_correct_csv, index=False)
                print(f"Saved class {class_num} correct results to {class_correct_csv}")
            
            if class_incorrect_results:
                class_incorrect_df = pd.DataFrame(class_incorrect_results)
                class_incorrect_csv = os.path.join(checkpoint_output_dir, f'class_{class_num}_incorrect_rsv_gi.csv')
                class_incorrect_df.to_csv(class_incorrect_csv, index=False)
                print(f"Saved class {class_num} incorrect results to {class_incorrect_csv}")
            
            # Update aggregated CSV files
            if all_correct_results:
                all_correct_df = pd.DataFrame(all_correct_results)
                all_correct_csv = os.path.join(checkpoint_output_dir, 'correct_rsv_gi_analysis.csv')
                all_correct_df.to_csv(all_correct_csv, index=False)
                print(f"Updated aggregated correct results including class {class_num}")
            
            if all_incorrect_results:
                all_incorrect_df = pd.DataFrame(all_incorrect_results)
                all_incorrect_csv = os.path.join(checkpoint_output_dir, 'incorrect_rsv_gi_analysis.csv')
                all_incorrect_df.to_csv(all_incorrect_csv, index=False)
                print(f"Updated aggregated incorrect results including class {class_num}")
            
            # Create/update correlation analysis after each class
            if all_correct_results:
                all_correct_df = pd.DataFrame(all_correct_results)
                summary_path = os.path.join(checkpoint_output_dir, 'correct_summary.csv')
                
                # Collect correlation data
                correlations = []
                total_gi_cols = [col for col in all_correct_df.columns if col.startswith('total_gi_')]
                for i in range(len(total_gi_cols)):
                    sv_column = f'singular_value_{i}'
                    gi_column = f'total_gi_{i}'
                    
                    if sv_column in all_correct_df.columns and gi_column in all_correct_df.columns:
                        correlation = all_correct_df[[sv_column, gi_column]].corr().iloc[0, 1]
                        correlations.append({
                            'rsv_index': i,
                            'avg_singular_value': all_correct_df[sv_column].mean(),
                            'avg_total_gi': all_correct_df[gi_column].mean(),
                            'correlation': correlation,
                            'classes_processed': class_num + 1  # Track progress
                        })
                
                corr_df = pd.DataFrame(correlations)
                corr_df.to_csv(summary_path, index=False)
                print(f"Updated correlation summary with class {class_num} at {summary_path}")
            
            # Create a progress summary file
            summary_data = {
                'checkpoint': checkpoint_name,
                'classes_completed': class_num + 1,  # Classes processed so far
                'correct_samples': len(all_correct_results),
                'incorrect_samples': len(all_incorrect_results),
            }
            
            if all_correct_results:
                all_correct_df = pd.DataFrame(all_correct_results)
                summary_data.update({
                    'correct_mean_pos_gi': all_correct_df['mean_pos_gi'].mean(),
                    'correct_mean_neg_gi': all_correct_df['mean_neg_gi'].mean(), 
                    'correct_mean_total_gi': all_correct_df['mean_total_gi'].mean(),
                    'correct_percent_pos_gi_above_1': all_correct_df['percent_pos_gi_above_1'].mean(),
                    'correct_percent_neg_gi_above_1': all_correct_df['percent_neg_gi_above_1'].mean(),
                    'correct_percent_total_gi_above_1': all_correct_df['percent_total_gi_above_1'].mean(),
                })
            
            if all_incorrect_results:
                all_incorrect_df = pd.DataFrame(all_incorrect_results)
                summary_data.update({
                    'incorrect_mean_pos_gi': all_incorrect_df['mean_pos_gi'].mean(),
                    'incorrect_mean_neg_gi': all_incorrect_df['mean_neg_gi'].mean(),
                    'incorrect_mean_total_gi': all_incorrect_df['mean_total_gi'].mean(),
                    'incorrect_percent_pos_gi_above_1': all_incorrect_df['percent_pos_gi_above_1'].mean(),
                    'incorrect_percent_neg_gi_above_1': all_incorrect_df['percent_neg_gi_above_1'].mean(),
                    'incorrect_percent_total_gi_above_1': all_incorrect_df['percent_total_gi_above_1'].mean(),
                })
            
            progress_df = pd.DataFrame([summary_data])
            progress_path = os.path.join(checkpoint_output_dir, 'rsv_gi_progress.csv')
            progress_df.to_csv(progress_path, index=False)
            print(f"Updated progress summary with class {class_num} at {progress_path}")
        
        print(f"\nCompleted processing checkpoint: {checkpoint_name}")
    
    print("All checkpoints processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GI along RSV directions and save to CSV")
    parser.add_argument('--config', required=True, help='Path to experiment configuration file')
    
    args = parser.parse_args()
    process_experiments(args.config)
