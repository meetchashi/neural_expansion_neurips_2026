import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import glob
import argparse

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Import the optimized compute_GIs_for_sample function
from src.analysis.gi_calculation import compute_GIs_for_sample

def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(config):
    """Load dataset based on configuration."""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()
    data_dir = os.path.expanduser(dataset_config['data_dir'])
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = MNIST(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = MNIST(root=data_dir, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset

def load_model_from_config(config, checkpoint_path, device):
    """Load model based on configuration and checkpoint."""
    model_config_path = os.path.expanduser(config['model']['config_path'])
    
    # Load the model configuration
    with open(model_config_path, 'r') as f:
        model_yaml = yaml.safe_load(f)
    
    # Extract information from the model configuration
    model_name = model_yaml['model']['name']
    model_module = model_yaml['model'].get('module', 'src.models')
    model_file = model_yaml['model'].get('file', 'mlp')
    model_args = model_yaml['model'].get('args', {})
    
    # Get number of classes from dataset
    num_classes = model_yaml['dataset'].get('num_classes', 10)
    
    # Import the module containing the model
    import importlib
    try:
        module = importlib.import_module(f"{model_module}.{model_file}")
    except ImportError as e:
        print(f"Error importing module {model_module}.{model_file}: {e}")
        raise
    
    # Get the model class and instantiate it
    try:
        model_class = getattr(module, model_name)
    except AttributeError as e:
        print(f"Error: Class {model_name} not found in module {model_module}.{model_file}")
        raise
    
    # For MLP, add input_dim and output_dim if not included in args
    if model_name == "PlainNN":
        if 'input_dim' not in model_args:
            model_args['input_dim'] = 28*28  # Default for MNIST
        if 'output_dim' not in model_args:
            model_args['output_dim'] = num_classes
    
    # Instantiate the model
    model = model_class(**model_args)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def get_checkpoint_paths(config):
    """Get list of checkpoint paths based on configuration."""
    checkpoint_config = config['checkpoints']
    strategy = checkpoint_config['strategy']
    base_dir = os.path.expanduser(checkpoint_config['base_dir'])
    
    if strategy == 'linear':
        start_cp = checkpoint_config['start_checkpoint']
        end_cp = checkpoint_config['end_checkpoint']
        num_checkpoints = checkpoint_config['num_checkpoints']
        
        # Get all checkpoints
        all_checkpoints = sorted(glob.glob(os.path.join(base_dir, "model_epoch*.pth")))
        
        # Find indices for start and end checkpoints
        start_idx = None
        end_idx = None
        
        for i, cp in enumerate(all_checkpoints):
            cp_name = os.path.basename(cp)
            if cp_name == start_cp:
                start_idx = i
            if cp_name == end_cp:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            raise ValueError(f"Couldn't find start or end checkpoint in {base_dir}")
        
        # Calculate intermediate checkpoints
        if num_checkpoints <= 2:
            selected_checkpoints = [all_checkpoints[start_idx], all_checkpoints[end_idx]]
        else:
            indices = np.linspace(start_idx, end_idx, num_checkpoints, dtype=int)
            selected_checkpoints = [all_checkpoints[i] for i in indices]
        
        return selected_checkpoints
    
    elif strategy == 'custom':
        return [os.path.join(base_dir, cp) for cp in checkpoint_config['custom_checkpoints']]
    
    elif strategy == 'pattern':
        checkpoints = []
        for pattern in checkpoint_config['patterns']:
            matched = glob.glob(os.path.join(base_dir, pattern))
            checkpoints.extend(matched)
        return sorted(checkpoints)
    
    else:
        raise ValueError(f"Unsupported checkpoint strategy: {strategy}")

def cluster_samples_by_class(dataset, num_classes=10):
    """Group dataset indices by their class labels."""
    clusters = {i: [] for i in range(num_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        clusters[label].append(idx)
    return clusters

# Fix in analyze_gi_for_checkpoint function
def analyze_gi_for_checkpoint(config, checkpoint_path, device):
    """
    Analyze GIs between test samples and training samples for a given checkpoint.
    Save results to CSV files after processing each class.
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Get checkpoint name for output files
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    
    # Load datasets
    train_dataset, test_dataset = load_dataset(config)
    
    # Load model from the current checkpoint
    model = load_model_from_config(config, checkpoint_path, device)
    
    # Group samples by class
    train_by_class = cluster_samples_by_class(train_dataset, 10)  # Assuming 10 classes for MNIST
    
    # Get GI analysis parameters
    gi_config = config['gi_analysis']
    max_alpha = gi_config.get('max_alpha', 3.0)
    step_size = gi_config.get('step_size', 0.01)
    batch_size = gi_config.get('batch_size', 6000)  # Add batch size parameter
    normalized = gi_config.get('normalized', False)  # Whether to normalize directions
    
    # Get sample selection config
    sample_config = config['samples']
    k_correct = sample_config.get('k_correct', 5)
    k_incorrect = sample_config.get('k_incorrect', 5)
    
    # Get classes to analyze
    classes_to_analyze = config['dataset'].get('classes', list(range(10)))
    
    # Create output directory
    output_dir = os.path.join(config['gi_analysis']['analysis_output_dir'], checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize empty lists to track all results
    all_correct_results = []
    all_incorrect_results = []
    
    # Load SVD data from the structure in the first code
    svd_data_root = config['output_dir']
    
    print(f"Processing {len(classes_to_analyze)} classes...")
    
    # For each class
    for class_idx in classes_to_analyze:
        print(f"\nProcessing class {class_idx}...")
        
        # Initialize class-specific results
        correct_results = []
        incorrect_results = []
        
        # Paths to class directories
        class_dir = os.path.join(svd_data_root, f"class_{class_idx}")
        
        # Process correctly classified samples
        correct_sample_dirs = []
        for sample_dir in glob.glob(os.path.join(class_dir, "correct/sample_*")):
            correct_sample_dirs.append(sample_dir)
        
        # Select k_correct samples (or all if fewer)
        if k_correct is not None and k_correct < len(correct_sample_dirs):
            correct_sample_dirs = np.random.choice(correct_sample_dirs, size=k_correct, replace=False)
        
        # Process incorrectly classified samples
        incorrect_sample_dirs = []
        for sample_dir in glob.glob(os.path.join(class_dir, "incorrect/sample_*")):
            incorrect_sample_dirs.append(sample_dir)
        
        # Select k_incorrect samples (or all if fewer)
        if k_incorrect is not None and k_incorrect < len(incorrect_sample_dirs):
            incorrect_sample_dirs = np.random.choice(incorrect_sample_dirs, size=k_incorrect, replace=False)
        
        # Process all samples
        for sample_dirs, results_list, is_correct, all_results_list in [
            (correct_sample_dirs, correct_results, True, all_correct_results),
            (incorrect_sample_dirs, incorrect_results, False, all_incorrect_results)
        ]:
            for sample_dir in tqdm(sample_dirs, desc=f"Class {class_idx} {'correct' if is_correct else 'incorrect'}"):
                # Extract sample index from directory name
                sample_idx = int(os.path.basename(sample_dir).split('_')[1])
                
                # Load metadata from any checkpoint dir, since sample info doesn't change
                for cp_name in os.listdir(sample_dir):
                    cp_dir_path = os.path.join(sample_dir, cp_name)
                    if os.path.isdir(cp_dir_path):
                        metadata_path = os.path.join(cp_dir_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            break
                else:
                    print(f"Warning: No metadata found for sample {sample_idx}, skipping")
                    continue
                
                true_class = metadata['true_class']
                
                # Get the test sample
                test_sample, _ = train_dataset[sample_idx]  # Using train dataset as specified in config
                
                # Run the model to get the current prediction with the loaded checkpoint
                with torch.no_grad():
                    test_input = test_sample.unsqueeze(0).to(device)
                    outputs = model(test_input)
                    predicted_class = outputs.argmax(dim=1).item()
                
                # Get all training samples of the predicted class
                train_indices = train_by_class[predicted_class]
                
                # To avoid self-comparison, remove the sample itself from indices if needed
                if sample_idx in train_indices:
                    train_indices.remove(sample_idx)
                
                # Stack training samples
                train_samples = torch.stack([train_dataset[i][0] for i in train_indices]).to(device)
                
                # Calculate GIs using the optimized function
                positive_gis, negative_gis = compute_GIs_for_sample(
                    model=model,
                    sample=test_sample,
                    class_samples=train_samples,
                    correct_class=predicted_class,
                    normalized=normalized,
                    step_size=step_size,
                    max_eps=max_alpha,
                    device=device,
                    batch_size=batch_size
                )
                
                # Convert numpy arrays to lists for JSON serialization
                positive_gis = positive_gis.tolist() if isinstance(positive_gis, np.ndarray) else positive_gis
                
                # Calculate statistics
                num_gi_above_1 = sum(v > 1.0 for v in positive_gis)
                percent_gi_above_1 = 100 * num_gi_above_1 / len(positive_gis) if len(positive_gis) else 0
                
                # Create result dictionary
                result = {
                    'sample_idx': sample_idx,
                    'class': class_idx,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'is_correct': predicted_class == true_class,
                    'checkpoint': checkpoint_name,  # Add checkpoint name to result
                    'num_train_samples': len(train_indices),
                    'percent_gi_above_1': percent_gi_above_1,
                    'min_gi': min(positive_gis) if len(positive_gis) else None,
                    'max_gi': max(positive_gis) if len(positive_gis) else None,
                    'mean_gi': np.mean(positive_gis) if len(positive_gis) else None,
                    'std_gi': np.std(positive_gis) if len(positive_gis) else None,
                }
                
                # Add individual GIs
                for i, gi in enumerate(positive_gis):
                    result[f'gi_{i}'] = gi
                
                # Add to appropriate results list
                results_list.append(result)
                all_results_list.append(result)
        
        # Save class-specific results to CSV
        if correct_results:
            class_correct_df = pd.DataFrame(correct_results)
            class_correct_csv = os.path.join(output_dir, f'class_{class_idx}_correct_gi_analysis.csv')
            class_correct_df.to_csv(class_correct_csv, index=False)
            print(f"Saved correctly classified results for class {class_idx} to {class_correct_csv}")
        
        if incorrect_results:
            class_incorrect_df = pd.DataFrame(incorrect_results)
            class_incorrect_csv = os.path.join(output_dir, f'class_{class_idx}_incorrect_gi_analysis.csv')
            class_incorrect_df.to_csv(class_incorrect_csv, index=False)
            print(f"Saved incorrectly classified results for class {class_idx} to {class_incorrect_csv}")
        
        # Update the aggregated CSVs (all classes processed so far)
        if all_correct_results:
            all_correct_df = pd.DataFrame(all_correct_results)
            all_correct_csv = os.path.join(output_dir, f'correct_gi_analysis.csv')
            all_correct_df.to_csv(all_correct_csv, index=False)
            print(f"Updated aggregated correctly classified results with class {class_idx}")
        
        if all_incorrect_results:
            all_incorrect_df = pd.DataFrame(all_incorrect_results)
            all_incorrect_csv = os.path.join(output_dir, f'incorrect_gi_analysis.csv')
            all_incorrect_df.to_csv(all_incorrect_csv, index=False)
            print(f"Updated aggregated incorrectly classified results with class {class_idx}")
def get_checkpoint_paths_from_directory(config):
    """Get list of checkpoint paths by scanning experiment directory structure."""
    root_dir = os.path.expanduser(config['output_dir'])
    if not os.path.exists(root_dir):
        raise ValueError(f"Experiment directory does not exist: {root_dir}")
    
    # Find all unique checkpoints by scanning directory structure
    checkpoint_names = set()
    
    # Scan through class directories
    for class_dir in os.listdir(root_dir):
        if not class_dir.startswith('class_'):
            continue
        
        class_path = os.path.join(root_dir, class_dir)
        # Scan through correctness directories
        for correctness in ['correct', 'incorrect']:
            correctness_path = os.path.join(class_path, correctness)
            if not os.path.exists(correctness_path):
                continue
            
            # Scan through sample directories
            for sample_dir in os.listdir(correctness_path):
                if not sample_dir.startswith('sample_'):
                    continue
                
                # Scan through epoch directories
                sample_path = os.path.join(correctness_path, sample_dir)
                for epoch_dir in os.listdir(sample_path):
                    if epoch_dir.startswith('model_epoch'):
                        # Look for metadata file to get checkpoint path
                        metadata_path = os.path.join(sample_path, epoch_dir, "metadata.json")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                # Add the actual checkpoint path to our set
                                if 'checkpoint' in metadata:
                                    checkpoint_names.add(metadata['checkpoint'])
                            except Exception as e:
                                print(f"Error reading metadata from {metadata_path}: {e}")
    
    # Convert to sorted list
    checkpoint_paths = sorted(list(checkpoint_names))
    print(f"Found {len(checkpoint_paths)} unique checkpoints")
    
    return checkpoint_paths
def main():
    parser = argparse.ArgumentParser(description="Analyze GIs between samples for specified checkpoint(s)")
    parser.add_argument('--config', required=True, help='Path to experiment config YAML')
    # parser.add_argument('--use-directory', action='store_true', help='Discover checkpoints from directory structure')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['gi_analysis'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Get checkpoint paths
    # if args.use_directory:
    checkpoint_paths = get_checkpoint_paths_from_directory(config)
    # else:
    #     checkpoint_paths = get_checkpoint_paths(config)
    
    print(f"Found {len(checkpoint_paths)} checkpoints for analysis")
    print(checkpoint_paths)
    
    # Process each checkpoint
    for checkpoint_path in checkpoint_paths:
        analyze_gi_for_checkpoint(config, checkpoint_path, device)

if __name__ == "__main__":
    main()

'''
nohup python3 experiments/gi_jacobian_svd/get_gi_along_train_samples.py --config "configs/experiments/mlp_mnist_exp2.yaml" > get_gi_along_train_samples.log 2>&1 &
nohup python3 experiments/gi_jacobian_svd/get_gi_along_train_samples.py --config "configs/experiments/mlp_mnist_exp3.yaml" > get_gi_along_train_samples.log 2>&1 &
'''
