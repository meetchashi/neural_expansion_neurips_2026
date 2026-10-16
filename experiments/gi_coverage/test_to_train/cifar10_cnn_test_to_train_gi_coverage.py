import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from pathlib import Path
from src.analysis.gi_calculation import compute_GIs_for_sample

def get_cifar_image_by_index(dataset, index):
    return dataset[index][0]

def load_pretrained_model(model_name, device):
    """
    Load a pre-trained model for CIFAR-10
    
    Args:
        model_name: Name of the model to load ("resnet20" or "vgg16_bn")
        device: Device to load the model on
    """
    if model_name == "resnet20":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    elif model_name == "vgg16_bn":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet20' or 'vgg16_bn'")
    
    model = model.to(device)
    model.eval()
    return model

def cluster_train_by_labels(train_dataset, total_class=10):
    """Group training data indices by class labels"""
    clusters = {i: [] for i in range(total_class)}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        clusters[label].append(idx)
    return clusters


def calculate_gis_from_train_to_test(model, test_image, class_train_images, class_id, device, normalized=False):
    """Calculate generalization intervals from train to test samples"""
    pos_gis, neg_gis = compute_GIs_for_sample(
        model=model,
        sample=test_image,
        normalized=normalized,
        class_samples=class_train_images,
        correct_class=class_id,
        step_size=0.01,
        max_eps=3.0,
        device=device, 
        batch_size=2000
    )
    return pos_gis

def process_class_cifar10(class_id, model_name, data_path, output_path, k=None, plot=False, use_random_labels=False, total_class=10):
    """
    Process a specific class from CIFAR-10 to calculate generalization intervals
    
    Args:
        class_id: Class ID (0-9) to process
        model_name: Model architecture to use ("resnet20" or "vgg16_bn")
        data_path: Path to store/load CIFAR-10 dataset
        k: Number of test samples to randomly select (if None, use all)
        plot: Whether to plot histograms
        use_random_labels: Whether to use randomized labels
        total_class: Total number of classes in the dataset
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load CIFAR-10 datasets
    full_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    # Split off a validation set if needed
    val_size = int(0.2 * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # If using random labels, randomize the training labels
    if use_random_labels:
        # Create a copy of the dataset with randomized labels
        random_labels = np.random.randint(0, total_class, size=len(train_dataset))
        
        # Since datasets from random_split don't allow direct label modification,
        # we need to create a wrapper or modify the indices
        original_indices = train_dataset.indices
        original_dataset = train_dataset.dataset
        
        # Create a dictionary mapping from original indices to random labels
        random_label_map = {original_indices[i]: random_labels[i] for i in range(len(train_dataset))}
        
        # Create a wrapper class to override the __getitem__ method
        class RandomLabelDataset:
            def __init__(self, dataset, indices, label_map):
                self.dataset = dataset
                self.indices = indices
                self.label_map = label_map
            
            def __getitem__(self, idx):
                original_idx = self.indices[idx]
                img, _ = self.dataset[original_idx]
                return img, self.label_map[original_idx]
            
            def __len__(self):
                return len(self.indices)
        
        train_dataset = RandomLabelDataset(original_dataset, original_indices, random_label_map)
    
    # Count class distribution
    train_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]
    
    train_counts = {i: train_labels.count(i) for i in range(total_class)}
    test_counts = {i: test_labels.count(i) for i in range(total_class)}
    
    print("\nTrain class distribution:", train_counts)
    print("Test class distribution:", test_counts)
    
    # Cluster training data based on labels
    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=total_class)
    
    # Load the pre-trained model
    model = load_pretrained_model(model_name, device)
    
    # Select test samples of the given class
    test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == class_id]
    if k is not None:
        test_indices = np.random.choice(test_indices, size=min(k, len(test_indices)), replace=False)
    
    results = []
    all_gis = []
    misclassified_results = []
    
    for idx in tqdm(test_indices, desc=f'Processing test class {class_id} ({model_name})'):
        test_img, true_label = test_dataset[idx]
        
        test_img = test_img.to(device).unsqueeze(0)
        pred = torch.argmax(model(test_img)).item()
        
        train_indices_for_pred_class = train_indices_by_class.get(pred, [])
        
        if train_indices_for_pred_class:
            # For CIFAR-10, we need to handle the dataset properly
            if hasattr(train_dataset, '__getitem__'):
                # Regular dataset
                train_imgs = torch.stack([get_cifar_image_by_index(train_dataset, i) for i in train_indices_for_pred_class[:200]]).to(device)
            else:
                # RandomLabelDataset
                train_imgs = torch.stack([train_dataset.__getitem__(i)[0] for i in range(len(train_dataset)) 
                                        if train_dataset.__getitem__(i)[1] == pred][:200]).to(device)
            
            gis = calculate_gis_from_train_to_test(model, test_img.squeeze(0), train_imgs, pred, device)
            num_gi_above_1 = sum(v > 1.0 for v in gis)
            percent_gi_above_1 = 100 * num_gi_above_1 / len(gis)
            
            if pred == true_label:
                results.append({
                    'sample_id': idx,
                    'predicted_class': pred,
                    'percent_gi_above_1': percent_gi_above_1,
                    **{f'gi_{i}': v for i, v in enumerate(gis)}
                })
                all_gis.extend(gis)
            else:
                misclassified_results.append({
                    'sample_id': idx,
                    'true_label': true_label,
                    'predicted_class': pred,
                    'percent_gi_above_1': percent_gi_above_1,
                    **{f'gi_{i}': v for i, v in enumerate(gis)}
                })
    
    # Save results
    label_type = "random" if use_random_labels else "real"
    # Ensure parent output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    base_output_path = os.path.join(output_path, f'GI_cifar10_{label_type}_{model_name}_experiment_outputs')
    os.makedirs(base_output_path, exist_ok=True)
    
    out_csv = os.path.join(base_output_path, f'from_train_to_test_GIs_class_{class_id}.csv') if k is None else os.path.join(base_output_path, f'from_train_to_test_GIs_class_{class_id}_k{k}.csv')
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV for correctly classified samples → {out_csv}")
    
    # Save misclassified results in a single CSV
    if misclassified_results:
        misclassified_csv_path = os.path.join(
            base_output_path,
            f'misclassified_from_class_{class_id}.csv'
        )
        df_misclassified = pd.DataFrame(misclassified_results)
        df_misclassified.to_csv(misclassified_csv_path, index=False)
        print(f"Saved CSV for misclassified samples → {misclassified_csv_path}")
    
    # Only plot if the plot argument is True
    if plot and all_gis:
        plt.figure(figsize=(10, 6))
        plt.hist(all_gis, bins=30, color='skyblue', edgecolor='black')
        plt.title(f'GI Distribution for Class {class_id} - {model_name} ({label_type} labels)')
        plt.xlabel('Generalization Interval')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_path, f'GI_hist_class_{class_id}.png'))
        print(f"Saved histogram for class {class_id}")
    elif plot and not all_gis:
        print(f"No correctly classified samples for class {class_id} to plot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_id', type=int, required=True, help='Class ID (0-9) to process')
    parser.add_argument('--model_name', type=str, choices=['resnet20', 'vgg16_bn'], required=True, help='Model to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to store/load CIFAR-10 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--k', type=int, default=None, help='Randomly pick k test samples (default: all)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of histograms')
    parser.add_argument('--random_labels', action='store_true', help='Use randomized labels')
    parser.add_argument('--total_class', type=int, default=10, help='Total classes (default: 10)')
    args = parser.parse_args()
    
    process_class_cifar10(
        class_id=args.class_id,
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k,
        plot=args.plot,
        use_random_labels=args.random_labels,
        total_class=args.total_class
    )

'''
Example usage:

# With real labels, ResNet20, without plotting
python cifar10_cnn_test_to_train_gi_coverage.py \
  --class_id 5 \
  --model_name resnet20 \
  --data_path "~/Desktop/Research/My Projects/deep_learning_manifold_expansion/notebooks/data/CIFAR10" \
  --output_path "../../results/cifar_10_resnet_20" \
  --k 50


# With random labels, VGG16_BN, with plotting
python cifar10_gi_calculation.py \
  --class_id 3 \
  --model_name vgg16_bn \
  --data_path "./data" \
  --k 50 \
  --plot \
'''
