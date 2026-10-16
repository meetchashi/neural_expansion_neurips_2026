# linear_approximation_threshold_binary_search.py

import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import glob

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors with error handling."""
    similarity = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
    
    # Check for NaN or infinite values
    if torch.isnan(similarity) or torch.isinf(similarity):
        return float('-inf')  # Return negative infinity to fail the threshold check
    
    return similarity

def get_penultimate_features(model, x, device='cuda'):
    """
    Extract penultimate layer features from the model.
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'extract_features'):
            features = model.extract_features(x)
            # Get the penultimate layer features (the layer before output)
            return features[-2]
        else:
            raise NotImplementedError("Model doesn't have extract_features method")

def compute_model_output_and_linear_approx(model, x, jacobian, direction, alpha, device='cuda'):
    """
    Compute the model's penultimate features and their linear approximation.
    
    Args:
        model: Neural network model
        x: Input sample tensor
        jacobian: Precomputed Jacobian matrix of penultimate features w.r.t. input
        direction: Direction vector to perturb along
        alpha: Perturbation magnitude
        device: Computation device
    
    Returns:
        actual_output: The actual penultimate layer features for perturbed input
        linear_approx: The linear approximation of the penultimate features
    """
    model.eval()
    x = x.to(device)
    
    # Get base features
    base_features = get_penultimate_features(model, x, device)
    
    # Compute perturbed input and get actual features
    input_shape = x.shape
    x_flat = x.view(1, -1)
    perturbed_input = (x_flat + alpha * direction).view(*input_shape)
    
    with torch.no_grad():
        actual_features = get_penultimate_features(model, perturbed_input, device)
    
    # Compute linear approximation using the precomputed Jacobian
    jacobian_tensor = torch.tensor(jacobian, device=device)
    jv = torch.matmul(jacobian_tensor, direction)
    linear_approx = base_features.clone() + alpha * jv.unsqueeze(0)
    
    return actual_features, linear_approx

def binary_search_alpha_threshold(model, x, jacobian, direction, threshold=0.95, 
                                 min_alpha=0.0, max_alpha=111.0, tol=0.01, 
                                 max_iterations=20, device='cuda', verbose=False):
    """
    Use binary search to find the maximum alpha where cosine similarity stays above threshold.
    Added iteration limit to prevent infinite loops.
    """
    # Initialize search bounds
    low = min_alpha
    high = max_alpha
    
    # First check if max_alpha works
    actual_output, linear_approx = compute_model_output_and_linear_approx(
        model, x, jacobian, direction, max_alpha, device
    )
    similarity = compute_cosine_similarity(actual_output[0], linear_approx[0])
    
    if verbose:
        print(f"Initial check at alpha={max_alpha}: similarity={similarity:.4f}")
    
    # If similarity at max_alpha is already above threshold, return max_alpha
    if similarity >= threshold:
        if verbose:
            print(f"Max alpha {max_alpha} works! Returning it.")
        return max_alpha
    
    # Next check if min_alpha works
    actual_output, linear_approx = compute_model_output_and_linear_approx(
        model, x, jacobian, direction, min_alpha, device
    )
    similarity = compute_cosine_similarity(actual_output[0], linear_approx[0])
    
    if verbose:
        print(f"Check at min_alpha={min_alpha}: similarity={similarity:.4f}")
    
    # If min_alpha doesn't work, return 0
    if similarity < threshold:
        if verbose:
            print(f"Even min_alpha {min_alpha} doesn't work. Returning 0.")
        return 0.0
    
    # Binary search
    iterations = 0
    while high - low > tol and iterations < max_iterations:
        iterations += 1
        mid = (low + high) / 2
        
        if verbose:
            print(f"Iteration {iterations}: Testing alpha = {mid:.4f}")
        
        actual_output, linear_approx = compute_model_output_and_linear_approx(
            model, x, jacobian, direction, mid, device
        )
        similarity = compute_cosine_similarity(actual_output[0], linear_approx[0])
        
        if verbose:
            print(f"  Similarity: {similarity:.4f}")
        
        if similarity >= threshold:
            low = mid  # This alpha works, try larger
            if verbose:
                print(f"  Similarity above threshold. New range: [{low:.4f}, {high:.4f}]")
        else:
            high = mid  # This alpha is too large, try smaller
            if verbose:
                print(f"  Similarity below threshold. New range: [{low:.4f}, {high:.4f}]")
    
    if iterations >= max_iterations and verbose:
        print(f"Warning: Binary search reached maximum iterations ({max_iterations})")
    
    return low  # Return the largest alpha that works

def calculate_linear_approximation_thresholds_binary(model, x, jacobian, singular_vectors, 
                                                    threshold=0.95, max_alpha=111.0, 
                                                    batch_size=50, tol=0.01, max_iterations=20,
                                                    device='cuda', verbose=False):
    """
    Calculate the maximum alpha for each RSV direction using binary search.
    
    Args:
        model: Neural network model
        x: Input sample tensor [1, C, H, W]
        jacobian: Precomputed Jacobian matrix
        singular_vectors: Right singular vectors from SVD [input_dim, num_directions]
        threshold: Cosine similarity threshold (default: 0.95)
        max_alpha: Maximum alpha to test
        batch_size: Number of directions to process in one batch
        tol: Tolerance for binary search convergence
        max_iterations: Maximum binary search iterations
        device: Device to run computations on
        verbose: Whether to print detailed logs
        
    Returns:
        alpha_thresholds: Array of maximum alpha values for each direction
    """
    model.eval()
    x = x.to(device)
    num_directions = singular_vectors.shape[1]
    
    # Initialize alpha thresholds
    alpha_thresholds = np.zeros(num_directions)
    
    # Process directions in batches to limit memory usage
    for batch_start in tqdm(range(0, num_directions, batch_size), desc="Processing RSV directions"):
        batch_end = min(batch_start + batch_size, num_directions)
        
        # Get current batch of singular vectors and normalize them
        batch_vectors = torch.tensor(singular_vectors[:, batch_start:batch_end], device=device)
        
        # Process each direction in the batch
        for i in tqdm(range(batch_end - batch_start)):
            direction = batch_vectors[:, i]
            direction = direction / torch.norm(direction)  # Normalize direction
            
            if verbose:
                print(f"\nProcessing direction {batch_start + i}")
            
            alpha_thresholds[batch_start + i] = binary_search_alpha_threshold(
                model=model,
                x=x,
                jacobian=jacobian,
                direction=direction,
                threshold=threshold,
                min_alpha=0.0,
                max_alpha=max_alpha,
                tol=tol,
                max_iterations=max_iterations,
                device=device,
                verbose=verbose
            )
    
    return alpha_thresholds

def load_svd_data(checkpoint_dir):
    """
    Load SVD data from either a single NPZ file or individual NPY files.
    Returns jacobian, V (singular vectors), and S (singular values).
    """
    # First try to load from NPZ file
    svd_data_npz_path = os.path.join(checkpoint_dir, "svd_data.npz")
    if os.path.exists(svd_data_npz_path):
        svd_data = np.load(svd_data_npz_path)
        jacobian = svd_data['jacobian'] 
        V = svd_data['V']
        S = svd_data['S']
        return jacobian, V, S
    
    # If NPZ not found, try individual NPY files
    jacobian_path = os.path.join(checkpoint_dir, "jacobian.npy")
    svd_V_path = os.path.join(checkpoint_dir, "svd_V.npy")
    svd_S_path = os.path.join(checkpoint_dir, "svd_S.npy")
    
    if not os.path.exists(jacobian_path) or not os.path.exists(svd_V_path) or not os.path.exists(svd_S_path):
        raise FileNotFoundError(f"SVD data files not found in {checkpoint_dir}")
    
    jacobian = np.load(jacobian_path)
    V = np.load(svd_V_path)
    S = np.load(svd_S_path)
    
    return jacobian, V, S

def process_sample(sample_dir, checkpoint, config, device='cuda'):
    """Process a single sample for a checkpoint."""
    # Load sample data
    sample_idx = int(os.path.basename(sample_dir).split('_')[-1])
    sample_data_path = os.path.join(sample_dir, f"sample_{sample_idx}_data.npy")
    if not os.path.exists(sample_data_path):
        print(f"Sample data not found: {sample_data_path}")
        return
    
    sample_data = torch.tensor(np.load(sample_data_path), device=device)
    
    # Load checkpoint-specific SVD data
    checkpoint_dir = os.path.join(sample_dir, checkpoint)
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    try:
        # Try to load SVD data from either NPZ or individual NPY files
        jacobian, V, S = load_svd_data(checkpoint_dir)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load model
    model_config_path = config['model']['config_path']
    checkpoint_path = os.path.join(config['checkpoints']['base_dir'], f"{checkpoint}.pth")
    
    # Dynamically load model based on config
    model = load_model_from_checkpoint(model_config_path, checkpoint_path, device)
    
    # Get verbose flag from config
    verbose = config['linear_approximation'].get('verbose', False)
    
    # Calculate linear approximation thresholds using binary search
    alphas = calculate_linear_approximation_thresholds_binary(
        model=model,
        x=sample_data,
        jacobian=jacobian,
        singular_vectors=V,
        threshold=config['linear_approximation']['threshold'],
        max_alpha=config['linear_approximation']['max_alpha'],
        batch_size=config['linear_approximation']['batch_size'],
        tol=config['linear_approximation']['tolerance'],
        max_iterations=config['linear_approximation'].get('max_iterations', 20),
        device=device,
        verbose=verbose
    )
    
    # Save results
    results = {
        'sample_idx': sample_idx,
        'checkpoint': checkpoint,
        'threshold': config['linear_approximation']['threshold'],
        'alpha_values': alphas.tolist(),
        'singular_values': S.tolist(),
    }
    
    output_path = os.path.join(checkpoint_dir, "linear_approximation_thresholds.npz")
    np.savez_compressed(output_path, alphas=alphas, S=S)
    
    # Also save a readable JSON summary
    json_path = os.path.join(checkpoint_dir, "linear_approximation_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def load_model_from_checkpoint(config_path, checkpoint_path, device='cuda'):
    """Load model from checkpoint path."""
    # Load the model config
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Get model architecture details
    model_name = model_config['model']['name']
    model_args = model_config['model'].get('args', {})
    
    # Import the module containing the model
    module_path = model_config['model'].get('module', 'src.models')
    model_file = model_config['model'].get('file', model_name.lower())
    
    # Add the project root to system path if needed
    project_root = Path(config_path).parents[2]  # Assuming config is in configs/training/
    import sys
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Import the module
    import importlib
    module = importlib.import_module(f"{module_path}.{model_file}")
    
    # Get the model class and instantiate it
    model_class = getattr(module, model_name)
    model = model_class(**model_args).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model

def run_linear_approximation_analysis(config_path):
    """Run linear approximation analysis for all samples and checkpoints."""
    config = load_config(config_path)
    
    # Add linear approximation section if not present
    if 'linear_approximation' not in config:
        config['linear_approximation'] = {
            'threshold': 0.95,
            'max_alpha': 111.0,
            'batch_size': 50,
            'tolerance': 0.01,
            'max_iterations': 20,
            'verbose': False,
        }
    
    from src.runtime import device_for
    device = device_for(config['gi_analysis'].get('device', 'auto'))
    print(f"Using device: {device}")
    
    # Get list of checkpoints to analyze
    checkpoints = config['gi_analysis']['analyze_checkpoints']
    if not checkpoints:
        print("No checkpoints specified for analysis. Please add checkpoints to analyze_checkpoints in config.")
        return
    
    # Get all samples directories
    output_dir = config['output_dir']
    sample_dirs = []
    for class_idx in config['dataset'].get('classes', range(10)):
        class_dir = os.path.join(output_dir, f"class_{class_idx}")
        for label_type in ['correct', 'incorrect']:
            type_dir = os.path.join(class_dir, label_type)
            if os.path.exists(type_dir):
                sample_dirs.extend(glob.glob(os.path.join(type_dir, "sample_*")))
    
    print(f"Found {len(sample_dirs)} samples to analyze")
    
    # Process all samples
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        for checkpoint in checkpoints:
            print(f"Processing sample {os.path.basename(sample_dir)} with checkpoint {checkpoint}")
            process_sample(sample_dir, checkpoint, config, device)
    
    print("\nLinear approximation analysis complete! Results saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Approximation Analysis for Neural Networks")
    parser.add_argument('--config', required=True, help='Path to experiment configuration file')
    args = parser.parse_args()
    
    run_linear_approximation_analysis(args.config)