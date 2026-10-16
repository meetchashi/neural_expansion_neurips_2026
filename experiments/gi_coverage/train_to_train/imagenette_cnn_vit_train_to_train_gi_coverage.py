import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tv_models
import timm
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from src.analysis.gi_calculation import compute_GIs_for_sample

# ImageNette to ImageNet class mapping
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # n01440764 → class index 0
    1: 217,  # n02102040 → class index 217
    2: 482,  # n02979186 → class index 482
    3: 491,  # n03000684 → class index 491
    4: 497,  # n03028079 → class index 497
    5: 566,  # n03394916 → class index 566
    6: 569,  # n03417042 → class index 569
    7: 571,  # n03425413 → class index 571
    8: 574,  # n03445777 → class index 574
    9: 701,  # n03888257 → class index 701
}

# Reverse mapping for convenience
IMAGENET_TO_IMAGENETTE = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}

def get_image_by_index(dataset, index):
    """Get image from dataset by index"""
    return dataset[index][0]

def load_pretrained_model(model_name, device):
    """
    Load a pre-trained model for ImageNet
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
    """
    if model_name == "VGG16":
        model = tv_models.vgg16(pretrained=True)
    elif model_name == "ResNet50":
        model = tv_models.resnet50(pretrained=True)
    elif model_name == "ViT-Base":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == "DeiT-Small":
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'VGG16', 'ResNet50', 'ViT-Base', or 'DeiT-Small'")
    
    model = model.to(device)
    model.eval()
    return model

def cluster_train_by_labels(train_dataset, class_indices=None):
    """Group training data indices by class labels"""
    clusters = {}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        if class_indices is None or label in class_indices:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
    return clusters

def calculate_gis_from_train_to_train(model, train_image, class_train_images, class_id, device, normalized=False):
    """Calculate generalization intervals from train to test samples"""
    pos_gis, neg_gis = compute_GIs_for_sample(
        model=model,
        sample=train_image,
        normalized=normalized,
        class_samples=class_train_images,
        correct_class=class_id,
        step_size=0.01,
        max_eps=3.0,
        device=device,
        batch_size=100  # Reduced for ImageNet-sized images
    )
    return pos_gis

def process_imagenette_class(class_id, model_name, data_path, output_path, k=None, plot=False, use_random_labels=False):
    """
    Process a specific ImageNette class to calculate generalization intervals
    
    Args:
        class_id: ImageNette class ID (0-9) to process
        model_name: Model architecture to use
        data_path: Path to ImageNette dataset
        output_path: Path for output files
        k: Number of test samples to randomly select (if None, use all)
        plot: Whether to plot histograms
        use_random_labels: Whether to use randomized labels
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNette datasets
    train_dir = os.path.join(data_path, "train")
    # val_dir = os.path.join(data_path, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(train_dir, transform=transform) #reuse the train dataset for validation
    
    # Map local ImageNette class ID to ImageNet class ID
    imagenet_class_id = IMAGENETTE_TO_IMAGENET[class_id]
    
    # If using random labels, randomize the training labels
    if use_random_labels:
        # Create random labels for training data
        np.random.seed(42)  # For reproducibility
        imagenette_classes = list(IMAGENETTE_TO_IMAGENET.keys())
        random_labels = np.random.choice(imagenette_classes, size=len(train_dataset))
        
        # Create a wrapper class to override the __getitem__ method
        class RandomLabelDataset:
            def __init__(self, dataset, random_labels):
                self.dataset = dataset
                self.random_labels = random_labels
            
            def __getitem__(self, idx):
                img, _ = self.dataset[idx]
                return img, self.random_labels[idx]
            
            def __len__(self):
                return len(self.dataset)
        
        train_dataset = RandomLabelDataset(train_dataset, random_labels)
    
    # Count class distribution
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    
    imagenette_classes = list(IMAGENETTE_TO_IMAGENET.keys())
    train_counts = {i: train_labels.count(i) for i in imagenette_classes}
    val_counts = {i: val_labels.count(i) for i in imagenette_classes}
    
    print("\nTrain class distribution:", train_counts)
    print("Val class distribution:", val_counts)
    
    # Cluster training data based on labels
    train_indices_by_class = cluster_train_by_labels(train_dataset, class_indices=imagenette_classes)
    
    # Load the pre-trained model
    model = load_pretrained_model(model_name, device)
    
    # Select validation samples of the given class
    val_indices = [i for i in range(len(val_dataset)) if val_dataset[i][1] == class_id]
    if k is not None:
        val_indices = np.random.choice(val_indices, size=min(k, len(val_indices)), replace=False)
    
    results = []
    all_gis = []
    misclassified_results = []
    
    for idx in tqdm(val_indices, desc=f'Processing validation class {class_id} ({model_name})'):
        val_img, true_label = val_dataset[idx]
        
        val_img = val_img.to(device).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            output = model(val_img)
            if isinstance(output, tuple):  # For DeiT
                output = output[0]
            pred = torch.argmax(output).item()
        
        # Check if prediction is one of the ImageNette classes (in ImageNet space)
        if pred not in IMAGENET_TO_IMAGENETTE:
            continue
        
        # Map the ImageNet prediction back to ImageNette space
        pred_imagenette = IMAGENET_TO_IMAGENETTE[pred]
        
        train_indices_for_pred_class = train_indices_by_class.get(pred_imagenette, [])
        
        if train_indices_for_pred_class:
            # Limit the number of training samples to check to save computation
            sample_indices = np.random.choice(
                train_indices_for_pred_class, 
                size=min(100, len(train_indices_for_pred_class)), 
                replace=False
            )
            
            train_imgs = torch.stack([get_image_by_index(train_dataset, i) for i in sample_indices]).to(device)
            
            gis = calculate_gis_from_train_to_train(model, val_img.squeeze(0), train_imgs, pred, device)
            num_gi_above_1 = sum(v > 1.0 for v in gis)
            percent_gi_above_1 = 100 * num_gi_above_1 / len(gis)
            
            if pred_imagenette == true_label:
                results.append({
                    'sample_id': idx,
                    'predicted_class': pred_imagenette,
                    'predicted_imagenet_class': pred,
                    'percent_gi_above_1': percent_gi_above_1,
                    **{f'gi_{i}': v for i, v in enumerate(gis)}
                })
                all_gis.extend(gis)
            else:
                misclassified_results.append({
                    'sample_id': idx,
                    'true_label': true_label,
                    'true_imagenet_label': IMAGENETTE_TO_IMAGENET[true_label],
                    'predicted_class': pred_imagenette,
                    'predicted_imagenet_class': pred,
                    'percent_gi_above_1': percent_gi_above_1,
                    **{f'gi_{i}': v for i, v in enumerate(gis)}
                })
    
    # Save results
    label_type = "random" if use_random_labels else "real"
    # Ensure parent output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    base_output_path = os.path.join(output_path, f'GI_imagenette_{label_type}_{model_name}_experiment_outputs')
    os.makedirs(base_output_path, exist_ok=True)
    
    out_csv = os.path.join(base_output_path, f'from_train_to_train_GIs_class_{class_id}.csv') if k is None else os.path.join(base_output_path, f'from_train_to_train_GIs_class_{class_id}_k{k}.csv')
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved CSV for correctly classified samples → {out_csv}")
    else:
        print(f"\nNo correctly classified samples for class {class_id}")
    
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
    parser.add_argument('--class_id', type=int, required=True, help='ImageNette Class ID (0-9) to process')
    parser.add_argument('--model_name', type=str, choices=['VGG16', 'ResNet50', 'ViT-Base', 'DeiT-Small'], required=True, help='Model to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNette dataset containing train and val folders')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--k', type=int, default=None, help='Randomly pick k validation samples (default: all)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of histograms')
    parser.add_argument('--random_labels', action='store_true', help='Use randomized labels')
    args = parser.parse_args()
    
    process_imagenette_class(
        class_id=args.class_id,
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k,
        plot=args.plot,
        use_random_labels=args.random_labels
    )

'''
Example usage:

# With real labels, ResNet50, process 20 samples
python imagenette_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 3 \
  --model_name ResNet50 \
  --data_path "~/Desktop/Research/My Projects/central_datasets/imagenette_v1" \
  --output_path "../../results/imagenette_resnet50" \
  --k 20 \
  --plot
# With real labels, VGG16, process 20 samples
python imagenette_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 3 \
  --model_name VGG16 \
  --data_path "~/Desktop/Research/My Projects/central_datasets/imagenette_v1" \
  --output_path "../../results/imagenette_vgg16_random" \
  --k 20 \
  
# With ViT-Base model
python imagenette_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 3 \
  --model_name ViT-Base \
  --data_path "~/Desktop/Research/My Projects/central_datasets/imagenette_v1" \
  --output_path "../../results/imagenette_vit_base" \
  --k 20

# With DeiT-Small model
python imagenette_cnn_vit_test_to_train_gi_coverage.py \
  --class_id 3 \
  --model_name DeiT-Small \
  --data_path "~/Desktop/Research/My Projects/central_datasets/imagenette_v1" \
  --output_path "../../results/imagenette_deit_small" \
  --k 20


'''
