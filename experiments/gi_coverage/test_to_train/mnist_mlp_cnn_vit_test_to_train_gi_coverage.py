import os
os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib
matplotlib.use('Agg')

import torch
import os
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from src.analysis.gi_calculation import compute_GIs_for_sample
from pathlib import Path
import yaml
import importlib


def get_mnist_image_by_index(dataset, index):
    return dataset[index][0]


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


def cluster_train_by_labels(train_dataset, total_class=10):
    clusters = {i: [] for i in range(total_class)}
    for idx in range(len(train_dataset)):
        img, label = train_dataset[idx]
        clusters[label].append(idx)
    return clusters


def calculate_gis_from_test_to_train(model, test_image, class_train_images, class_id, device):
    pos_gis, _ = compute_GIs_for_sample(
        model=model,
        sample=test_image,
        normalized=False,
        class_samples=class_train_images,
        correct_class=class_id,
        step_size=0.01,
        max_eps=3.0,
        device=device
    )
    return pos_gis


def process_class(class_id, config_path, checkpoint_path, data_path, output_path, k=None, plot=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Processing class {class_id} with k={k}...")
    
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

    train_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]

    train_counts = {i: train_labels.count(i) for i in range(total_class)}
    test_counts = {i: test_labels.count(i) for i in range(total_class)}

    print("\nTrain class distribution:", train_counts)
    print("Test class distribution:", test_counts)  
    
    # Cluster training data based on labels
    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=total_class)

    # Select test samples of the given class
    test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == class_id]
    if k is not None:
        test_indices = np.random.choice(test_indices, size=k, replace=False)

    results = []
    all_gis = []
    misclassified_results = []

    for idx in tqdm(test_indices, desc=f'Processing test class {class_id}'):
        test_img, true_label = test_dataset[idx]
        if true_label != class_id:
            continue

        test_img = test_img.to(device).unsqueeze(0)
        pred = torch.argmax(model(test_img)).item()

        train_indices_for_pred_class = train_indices_by_class.get(pred, [])

        if train_indices_for_pred_class:
            train_imgs = torch.stack([get_mnist_image_by_index(train_dataset, i) for i in train_indices_for_pred_class]).to(device)
            gis = calculate_gis_from_test_to_train(model, test_img.squeeze(0), train_imgs, pred, device)
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
    base_output_path = os.path.join(output_path, f'GI_experiment_outputs_{model_type}')
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
        plt.title(f'GI Distribution for Class {class_id} ({model_type.upper()}, Correctly Classified)')
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
    parser.add_argument('--class_id', type=int, required=True, help='Class ID (0–9) to process')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MNIST dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--k', type=int, default=None, help='Randomly pick k test samples (default: all)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of histograms')
    args = parser.parse_args()

    process_class(
        class_id=args.class_id,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k,
        plot=args.plot
    )

'''
python3 experiments/gi_coverage/test_to_train/mnist_mlp_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 0 \
  --config ./configs/training/mnist_mlp.yaml \
  --checkpoint ./checkpoints/mnist_original_mlp/best_model_mnist.pth \
  --data_path ./data/MNIST-Original \
  --output_path ./experiments/results/temp/mnist_original_mlp \
  --plot

python3 experiments/gi_coverage/test_to_train/mnist_mlp_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 5 \
  --config ./configs/training/mnist_cnn.yaml \
  --checkpoint ./checkpoints/mnist_original_cnn/best_model_mnist.pth \
  --data_path ./data/MNIST-Original \
  --output_path ./experiments/results/temp/mnist_original_cnn \
  --k 50 \
  --plot

python3 experiments/gi_coverage/test_to_train/mnist_mlp_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 9 \
  --config ./configs/training/mnist_vit.yaml \
  --checkpoint ./checkpoints/mnist_original_vit/best_model_mnist.pth \
  --data_path ./data/MNIST-Original \
  --output_path ./experiments/results/temp/mnist_original_vit \
  --k 50 \
  --plot \
'''
