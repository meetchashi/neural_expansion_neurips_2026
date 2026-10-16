# ~/Desktop/Research/My Projects/neural_expansion/scripts/collect_svd_data.py

import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import glob
import json
from pathlib import Path
import importlib.util
import sys
import random

from torchvision import transforms
from torch.utils.data import DataLoader

def load_model_from_config(config_path, checkpoint_path):
    """
    Dynamically load a model from its config and weights.
    """
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

def load_dataset(dataset_name, data_dir, split='test'):
    """
    Load the dataset based on dataset name and split.
    """
    if dataset_name.lower() == 'mnist':
        from torchvision.datasets import MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if split == 'train':
            dataset = MNIST(root=data_dir, train=True, transform=transform, download=True)
        else:
            dataset = MNIST(root=data_dir, train=False, transform=transform, download=True)
    
    elif dataset_name.lower() == 'cifar10':
        from torchvision.datasets import CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if split == 'train':
            dataset = CIFAR10(root=data_dir, train=True, transform=transform, download=True)
        else:
            dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

def evaluate_model(model, dataset, device='cpu', batch_size=100):
    """
    Evaluate the model on the dataset and return predictions.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating model")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions
            for i in range(len(labels)):
                sample_idx = batch_idx * batch_size + i
                is_correct = (predicted[i] == labels[i]).item()
                
                all_predictions.append({
                    'sample_idx': sample_idx,
                    'true_label': labels[i].item(),
                    'predicted_label': predicted[i].item(),
                    'is_correct': is_correct
                })
    
    return all_predictions

def select_samples(predictions, class_idx, k, select_correct=True):
    """
    Select k samples from a specific class that are either correctly or incorrectly classified.
    """
    filtered_samples = []
    
    for pred in predictions:
        # Filter by class and correctness
        if pred['true_label'] == class_idx and pred['is_correct'] == select_correct:
            filtered_samples.append(pred)
    
    # If we have fewer than k samples, use all we have
    k = min(k, len(filtered_samples))
    
    if k == 0:
        print(f"Warning: No {'correctly' if select_correct else 'incorrectly'} classified samples found for class {class_idx}")
        return []
    
    # Randomly select k samples
    selected_samples = random.sample(filtered_samples, k)
    
    print(f"Selected {len(selected_samples)} {'correctly' if select_correct else 'incorrectly'} "
          f"classified samples for class {class_idx}")
    
    return selected_samples

def compute_jacobian(model, x, output_layer_idx=-2):
    """
    Compute the Jacobian of the penultimate layer features with respect to the input.
    """
    x.requires_grad_(True)
    
    # Function to extract features from a specific layer
    def get_penultimate_features(model, x, layer_idx):
        # This implementation will depend on the model architecture
        # For now, we'll assume a simple approach that works for many models
        if hasattr(model, 'extract_features'):
            features = model.extract_features(x)
            return features[layer_idx]
        
        # For models that don't have extract_features, we'd need to add hooks
        # This is a simplified example and may need to be adapted
        features = None
        for i, layer in enumerate(model.children()):
            x = layer(x)
            if i == layer_idx:
                features = x
        return features
    
    # Get features from the penultimate layer
    features = get_penultimate_features(model, x, output_layer_idx)
    
    # Initialize Jacobian matrix
    jacobian = torch.zeros(features.shape[1], x.numel())
    
    # Compute Jacobian one feature at a time
    for i in range(features.shape[1]):
        model.zero_grad()
        
        # Set all gradients to zero except for the current feature
        if i > 0:
            x.grad.zero_()
        
        # Backward pass for this feature
        features[0, i].backward(retain_graph=True)
        
        # Store the gradient
        jacobian[i] = x.grad.view(-1)
    
    # Reset requires_grad
    x.requires_grad_(False)
    
    return jacobian

def compute_svd(jacobian):
    """
    Compute the Singular Value Decomposition of the Jacobian matrix.
    Returns all right singular vectors even in the input space.
    """
    # Convert to NumPy for full SVD
    J_np = jacobian.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(J_np, full_matrices=True)
    V = Vt.T  # So V[:, i] is the i-th right singular vector
    
    return U, S, V

def analyze_sample(model, dataset, sample_idx, checkpoint_path, output_dir, device='cpu', return_outputs=False):
    x, y = dataset[sample_idx]
    x = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        pred_class = torch.argmax(output, 1).item()

    is_correct = pred_class == y
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]

    jacobian = compute_jacobian(model, x)
    U, S, V = compute_svd(jacobian)

    metadata = {
        "checkpoint": checkpoint_path,
        "sample_idx": sample_idx,
        "true_class": int(y),
        "predicted_class": int(pred_class),
        "is_correct": bool(is_correct),
        "svd_shape": {
            "U": U.shape,
            "S": S.shape,
            "V": V.shape
        }
    }

    if return_outputs:
        return jacobian, U, S, V, metadata

    os.makedirs(output_dir, exist_ok=True)
    
    # Save all arrays in a single NPZ file instead of individual NPY files
    np.savez_compressed(
        os.path.join(output_dir, "svd_data.npz"),
        jacobian=jacobian.cpu().numpy(),
        U=U,
        S=S,
        V=V
    )
    
    # Still save metadata as JSON for easy reading
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    return metadata

def get_checkpoints(config):
    """
    Get list of checkpoint paths based on the configuration.
    """
    base_dir = os.path.abspath(os.path.expanduser(config['checkpoints']['base_dir']))
    strategy = config['checkpoints']['strategy']

    if strategy == 'custom':
        checkpoint_paths = [os.path.abspath(os.path.join(base_dir, cp)) for cp in config['checkpoints']['custom_checkpoints']]

    elif strategy == 'linear':
        start_cp_path = os.path.abspath(os.path.join(base_dir, config['checkpoints']['start_checkpoint']))
        end_cp_path = os.path.abspath(os.path.join(base_dir, config['checkpoints']['end_checkpoint']))
        num_checkpoints = config['checkpoints']['num_checkpoints']

        # all_checkpoints = sorted([os.path.abspath(cp) for cp in glob.glob(os.path.join(base_dir, "model_epoch*.pth"))])

        import re

        def extract_epoch_batch(path):
            filename = os.path.basename(path)
            match = re.search(r"epoch(\d+)_batch(\d+)", filename)
            if match:
                return int(match.group(1)), int(match.group(2))
            return float('inf'), float('inf')  # sort unknowns last

        all_checkpoints = sorted(
            [os.path.abspath(cp) for cp in glob.glob(os.path.join(base_dir, "model_epoch*.pth"))],
            key=extract_epoch_batch
        )

        try:
            start_idx = all_checkpoints.index(start_cp_path)
            end_idx = all_checkpoints.index(end_cp_path)
        except ValueError as e:
            print("Available checkpoints:")
            for cp in all_checkpoints:
                print(" -", cp)
            raise ValueError(f"Checkpoint not found in directory: {e}")

        if num_checkpoints <= 1:
            checkpoint_paths = [all_checkpoints[start_idx]]
        elif num_checkpoints == 2:
            checkpoint_paths = [all_checkpoints[start_idx], all_checkpoints[end_idx]]
        else:
            indices = np.linspace(start_idx, end_idx, num_checkpoints, dtype=int)
            checkpoint_paths = [all_checkpoints[i] for i in sorted(set(indices))]

            if len(checkpoint_paths) < num_checkpoints:
                print(f"Warning: Found {len(checkpoint_paths)} unique checkpoints instead of {num_checkpoints}")

    elif strategy == 'pattern':
        checkpoint_paths = []
        for pattern in config['checkpoints']['patterns']:
            matched_paths = glob.glob(os.path.join(base_dir, pattern))
            checkpoint_paths.extend(matched_paths)
        checkpoint_paths = sorted(list(set([os.path.abspath(cp) for cp in checkpoint_paths])))

    else:
        raise ValueError(f"Unsupported checkpoint selection strategy: {strategy}")

    print(f"Selected {len(checkpoint_paths)} checkpoints")
    return checkpoint_paths



def run_data_collection(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    random.seed(config.get('seed', 42))
    output_dir = os.path.expanduser(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_config.yaml"), 'w') as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_paths = get_checkpoints(config)
    print(f"Found {len(checkpoint_paths)} checkpoints for analysis")

    dataset_config = config['dataset']
    dataset = load_dataset(
        dataset_name=dataset_config['name'],
        data_dir=os.path.expanduser(dataset_config['data_dir']),
        split=config['samples']['dataset_split']
    )

    classes_to_analyze = dataset_config.get('classes', list(range(10)))
    k_correct = config['samples']['k_correct']
    k_incorrect = config['samples']['k_incorrect']

    # Step 1: Use the LAST checkpoint to fix sample indices per class
    print("\nSelecting fixed samples using the last checkpoint")
    last_model = load_model_from_config(os.path.expanduser(config['model']['config_path']), checkpoint_paths[-1]).to(device)
    predictions = evaluate_model(last_model, dataset, device)

    sample_map = {}  # class_idx -> {'correct': [samples], 'incorrect': [samples]}
    for class_idx in classes_to_analyze:
        sample_map[class_idx] = {
            'correct': select_samples(predictions, class_idx, k_correct, True),
            'incorrect': select_samples(predictions, class_idx, k_incorrect, False)
        }

    # Step 2: For each sample per class, write all checkpoints directly under model_epoch folders
    for class_idx in classes_to_analyze:
        for label_type in ['correct', 'incorrect']:
            for sample in sample_map[class_idx][label_type]:
                sample_idx = sample['sample_idx']
                sample_dir = os.path.join(output_dir, f"class_{class_idx}", label_type, f"sample_{sample_idx}")
                os.makedirs(sample_dir, exist_ok=True)

                # Save the data only once if not already
                sample_data_path = os.path.join(sample_dir, f"sample_{sample_idx}_data.npy")
                if not os.path.exists(sample_data_path):
                    x, _ = dataset[sample_idx]
                    np.save(sample_data_path, x.unsqueeze(0).cpu().numpy())

                for checkpoint_path in checkpoint_paths:
                    print(f"Analyzing class {class_idx} sample {sample_idx} at {checkpoint_path}")
                    model = load_model_from_config(os.path.expanduser(config['model']['config_path']), checkpoint_path).to(device)
                    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
                    checkpoint_output_dir = os.path.join(sample_dir, checkpoint_name)
                    os.makedirs(checkpoint_output_dir, exist_ok=True)

                    # Save directly to checkpoint_output_dir without nesting again
                    jacobian, svd_u, svd_s, svd_v, meta = analyze_sample(
                        model=model,
                        dataset=dataset,
                        sample_idx=sample_idx,
                        checkpoint_path=checkpoint_path,
                        output_dir=None,  # unused now
                        device=device,
                        return_outputs=True
                    )

                    np.savez_compressed(
                        os.path.join(checkpoint_output_dir, "svd_data.npz"),
                        jacobian=jacobian.cpu().numpy(),
                        U=svd_u,
                        S=svd_s,
                        V=svd_v
                    )
                    with open(os.path.join(checkpoint_output_dir, "metadata.json"), 'w') as f:
                        json.dump(meta, f, indent=4)

    print("\nData collection complete! Results saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD Data Collection for GI Analysis")
    parser.add_argument('--config', required=True, help='Path to experiment configuration file')
    args = parser.parse_args()
    
    run_data_collection(args.config)







