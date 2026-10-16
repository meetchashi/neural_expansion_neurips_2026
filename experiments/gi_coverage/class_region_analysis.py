import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import yaml
import importlib
import argparse
from pathlib import Path

def load_model_from_config(config_path, checkpoint_path, device):
    """Load model based on configuration file."""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get model configuration
    model_config = config['model']
    model_name = model_config['name']
    model_args = model_config.get('args', {})
    
    # Import the module containing the model
    module_path = model_config.get('module', 'src.models')
    model_file = model_config.get('file', model_name.lower())
    
    try:
        # Import the module
        module = importlib.import_module(f"{module_path}.{model_file}")
        
        # Get the model class and instantiate it
        model_class = getattr(module, model_name)
        model = model_class(**model_args).to(device)
        
        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        return model, config
    except (ImportError, AttributeError) as e:
        print(f"Error loading model: {e}")
        raise

def get_mnist_image_by_index(dataset, index):
    return dataset[index][0]

def cluster_train_by_labels(train_dataset, total_class=10):
    clusters = {i: [] for i in range(total_class)}
    for idx in range(len(train_dataset)):
        img, label = train_dataset[idx]
        clusters[label].append(idx)
    return clusters

def calculate_gi(model, x1, x2, correct_class, step_size=0.01, max_eps=3.0, device="cpu"):
    """
    Calculate the generalization interval from x1 towards x2.
    Returns the epsilon at which prediction changes from correct_class.
    """
    model.eval()
    with torch.no_grad():
        # Ensure x1 and x2 are on the correct device
        x1, x2 = x1.to(device), x2.to(device)
        
        # Calculate direction (not normalized)
        direction = x2 - x1
        
        # Check initial prediction
        initial_output = model(x1.unsqueeze(0))
        initial_pred = torch.argmax(initial_output).item()
        
        # If initial prediction is not the correct class, GI is 0
        if initial_pred != correct_class:
            return 0.0
        
        # Step through the path
        for eps in np.arange(0, max_eps + step_size, step_size):
            # Calculate point at current epsilon
            point = x1 + eps * direction
            
            # Get model prediction
            output = model(point.unsqueeze(0))
            pred = torch.argmax(output).item()
            
            # Check if prediction changed
            if pred != correct_class:
                return eps
        
        # If prediction never changed, return max_eps
        return max_eps

def analyze_class_regions(model, start_point, end_point, device, step_size=0.01, max_eps=3.0):
    """
    Analyze how class predictions change along the path from start_point to end_point.
    Returns a list of intervals with their corresponding class predictions.
    """
    model.eval()
    with torch.no_grad():
        # Move samples to device
        start_point = start_point.to(device)
        end_point = end_point.to(device)
        
        # Calculate direction (not normalized)
        direction = end_point - start_point
        
        # Initialize variables
        current_class = None
        regions = []
        region_start = 0.0
        
        # Get initial class
        initial_output = model(start_point.unsqueeze(0))
        initial_class = torch.argmax(initial_output).item()
        current_class = initial_class
        
        # Step through the path
        steps = np.arange(0, max_eps + step_size, step_size)
        for eps in steps:
            # Calculate point at current epsilon
            point = start_point + eps * direction
            
            # Get model prediction
            output = model(point.unsqueeze(0))
            pred_class = torch.argmax(output).item()
            
            # Check if class changed
            if pred_class != current_class:
                # Record the previous region
                regions.append({
                    "start": region_start,
                    "end": eps,
                    "class": int(current_class)
                })
                
                # Start a new region
                region_start = eps
                current_class = pred_class
        
        # Add the final region
        regions.append({
            "start": region_start,
            "end": max_eps,
            "class": int(current_class)
        })
        
        return regions

def find_training_samples_with_low_gi(model, test_sample, test_label, train_dataset, train_indices, 
                                     device, max_samples=10, step_size=0.01, max_eps=3.0):
    """
    Find up to max_samples training samples with GI < 1 from test_sample to training sample.
    """
    low_gi_samples = []
    
    # Shuffle train indices to get random samples
    np.random.shuffle(train_indices)
    
    # Loop through training samples
    for train_idx in train_indices:
        if len(low_gi_samples) >= max_samples:
            break
            
        train_img = get_mnist_image_by_index(train_dataset, train_idx)
        
        # Calculate GI from test to train
        gi = calculate_gi(model, test_sample, train_img, test_label, step_size, max_eps, device)
        
        # If GI < 1, add to list
        if gi < 1.0:
            low_gi_samples.append({
                "train_idx": train_idx,
                "gi": gi
            })
    
    return low_gi_samples

def process_class_regions(class_id, config_path, checkpoint_path, data_path, output_path, k=None, max_eps=3.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing class {class_id}...")
    
    # Load model and config
    model, config = load_model_from_config(config_path, checkpoint_path, device)
    
    # Get model type for output naming
    model_type = config['model']['name'].lower()
    
    # Get dataset info from config
    dataset_config = config['dataset']
    total_class = dataset_config.get('num_classes', 10)
    
    # Define transform based on config
    transform_config = dataset_config.get('transform', {})
    transform_list = []
    if transform_config.get('to_tensor', True):
        transform_list.append(transforms.ToTensor())
    if 'normalize' in transform_config:
        mean = transform_config['normalize'].get('mean', (0.5,))
        std = transform_config['normalize'].get('std', (0.5,))
        transform_list.append(transforms.Normalize(mean, std))
    
    transform = transforms.Compose(transform_list)

    # Load datasets
    full_train = MNIST(root=data_path, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_path, train=False, transform=transform, download=True)
    val_size = int(dataset_config.get('val_split', 0.2) * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(config.get('seed', 42)))

    # Cluster training data based on labels
    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=total_class)

    # Select test samples of the given class
    test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == class_id]
    if k is not None:
        test_indices = np.random.choice(test_indices, size=k, replace=False)

    # Store results
    all_results = []
    test_sample_count = 0

    for test_idx in tqdm(test_indices, desc=f'Processing test samples of class {class_id}'):
        test_img, true_label = test_dataset[test_idx]
        test_img = test_img.to(device)
        
        # Find training samples with GI < 1 from test to train
        low_gi_samples = find_training_samples_with_low_gi(
            model, test_img, true_label,
            train_dataset, train_indices_by_class[class_id],
            device, max_samples=10, max_eps=max_eps
        )
        
        # If any low GI samples found, analyze class regions in the opposite direction
        if low_gi_samples:
            test_results = {
                "test_sample_idx": int(test_idx),
                "true_class": int(true_label),
                "train_samples": []
            }
            
            for sample in low_gi_samples:
                train_idx = sample["train_idx"]
                train_img = get_mnist_image_by_index(train_dataset, train_idx)
                
                # Analyze class regions from train to test (opposite direction)
                regions = analyze_class_regions(
                    model, train_img, test_img, device, 
                    step_size=0.01, max_eps=max_eps
                )
                
                test_results["train_samples"].append({
                    "train_idx": int(train_idx),
                    "gi_test_to_train": sample["gi"],
                    "regions": regions
                })
            
            all_results.append(test_results)
            test_sample_count += 1
    
    print(f"Found {test_sample_count} test samples with at least one training sample having GI < 1")
    
    # Save results to JSON
    base_output_path = os.path.join(output_path, f'class_regions_{model_type}')
    os.makedirs(base_output_path, exist_ok=True)
    
    output_file = os.path.join(base_output_path, f'train_to_test_regions_class_{class_id}.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved class region analysis to {output_file}")
    
    # Create visualizations for the first 5 test samples
    try:
        visualize_count = min(5, len(all_results))
        
        for i in range(visualize_count):
            test_result = all_results[i]
            test_idx = test_result["test_sample_idx"]
            train_samples = test_result["train_samples"]
            
            # Limit to at most 5 training samples per test sample for visualization
            vis_train_count = min(5, len(train_samples))
            
            plt.figure(figsize=(15, vis_train_count * 2))
            plt.suptitle(f"Class Regions from Train to Test Sample {test_idx} (Class {test_result['true_class']})", 
                        fontsize=16)
            
            for j in range(vis_train_count):
                train_sample = train_samples[j]
                train_idx = train_sample["train_idx"]
                regions = train_sample["regions"]
                
                plt.subplot(vis_train_count, 1, j+1)
                
                # Plot regions, correctly labeling the x-axis to show train → test direction
                for region in regions:
                    plt.barh(y=0, width=region["end"]-region["start"], left=region["start"], 
                            color=plt.cm.tab10(region["class"]), alpha=0.7)
                    
                    # Add text label if region is wide enough
                    if region["end"] - region["start"] > 0.2:
                        plt.text(
                            (region["start"] + region["end"]) / 2, 0,
                            f"Class {region['class']}",
                            ha='center', va='center'
                        )
                
                # Add vertical line at ε = 1.0
                plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
                plt.text(1.0, 0.5, 'ε = 1.0', color='red', ha='right', va='bottom', 
                        fontsize=10, rotation=90, transform=plt.gca().get_xaxis_transform())
                
                plt.yticks([])
                plt.xlim(0, max_eps)
                
                # Correctly label the x-axis to show it's train → test
                plt.xlabel('Distance along path (ε) from Training to Test Sample')
                
                # Add labels to start and end of axis to clarify direction
                plt.text(0, -0.2, f'Train {train_idx}', ha='left', va='top', 
                        transform=plt.gca().get_xaxis_transform())
                plt.text(max_eps, -0.2, f'Test {test_idx}', ha='right', va='top', 
                        transform=plt.gca().get_xaxis_transform())
                
                plt.title(f"Train {train_idx} → Test {test_idx} (GI test→train: {train_sample['gi_test_to_train']:.2f})")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(base_output_path, f'train_to_test_regions_vis_sample_{test_idx}.png'))
            plt.close()
        
        print(f"Saved visualizations for {visualize_count} test samples")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_id', type=int, required=True, help='Class ID (0–9) to process')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MNIST dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--k', type=int, default=None, help='Randomly pick k test samples (default: all)')
    parser.add_argument('--max_eps', type=float, default=3.0, help='Maximum epsilon for path analysis')
    args = parser.parse_args()

    process_class_regions(
        class_id=args.class_id,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k,
        max_eps=args.max_eps
    )

'''
python experiments/gi_coverage/class_region_analysis.py \
  --class_id 6 \
  --config ./configs/training/mnist_mlp.yaml \
  --checkpoint ./checkpoints/mnist_original_mlp/model_epoch0_batch0.pth \
  --data_path ./data/MNIST-Original \
  --output_path ./experiments/results \
  --k 100 \
  --max_eps 3.0
'''
