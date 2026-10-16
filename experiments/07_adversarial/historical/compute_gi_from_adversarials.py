import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets, models
import timm
from src.analysis.gi_calculation import compute_GIs_for_sample
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# ImageNet → ImageNette mapping
IMAGENET_TO_IMAGENETTE = {
    0: 0, 217: 1, 482: 2, 491: 3, 497: 4,
    566: 5, 569: 6, 571: 7, 574: 8, 701: 9
}

# --------- Utility Functions ----------

def load_model(model_arg, dataset_name, device):
    if dataset_name == "mnist":
        from src.models.mlp import PlainNN
        model = PlainNN(28*28, 1024, 512, 10)
        model.load_state_dict(torch.load(model_arg, map_location=device))
    elif dataset_name == "cifar10":
        if model_arg == "resnet20":
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        elif model_arg == "vgg16_bn":
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
        else:
            raise ValueError("CIFAR-10 supports only: resnet20, vgg16_bn")
    elif dataset_name == "imagenette":
        if model_arg == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_arg == "VGG16":
            model = models.vgg16(pretrained=True)
        elif model_arg == "ViT-Base":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
        elif model_arg == "DeiT-Small":
            model = timm.create_model("deit_small_patch16_224", pretrained=True)
        else:
            raise ValueError("ImageNette supports only: resnet50, VGG16, ViT-Base, DeiT-Small")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return model.to(device).eval()

def get_transform(dataset_name):
    if dataset_name == "mnist":
        return transforms.Compose([transforms.ToTensor()])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    elif dataset_name == "imagenette":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unknown dataset")

def load_train_dataset(dataset_name, data_path, transform):
    if dataset_name == "mnist":
        return datasets.MNIST(data_path, train=True, transform=transform, download=True)
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
    elif dataset_name == "imagenette":
        return datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
    else:
        raise NotImplementedError

# --------- GI Evaluation Logic ----------

def get_image_tensor(path, dataset_name):
    img = Image.open(path).convert("RGB")
    if dataset_name == "mnist":
        img = img.convert("L")
    transform = get_transform(dataset_name)
    return transform(img).unsqueeze(0)  # add batch dim

def organize_train_dataset_by_class(train_dataset):
    """Pre-organize dataset indices by class to avoid repeated scans"""
    print("Organizing training dataset by class...")
    
    if hasattr(train_dataset, 'targets'):
        # Fast path for datasets with .targets attribute (like MNIST/CIFAR)
        labels = train_dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
    elif hasattr(train_dataset, 'imgs'):
        # Fast path for ImageFolder datasets
        labels = [img[1] for img in train_dataset.imgs]
    else:
        # Slower fallback using DataLoader
        loader = DataLoader(train_dataset, batch_size=1000, num_workers=4)
        labels = []
        for _, batch_labels in tqdm(loader, desc="Collecting labels"):
            if isinstance(batch_labels, torch.Tensor):
                batch_labels = batch_labels.numpy()
            labels.extend(batch_labels)
    
    # Build class_to_indices dictionary
    class_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_to_indices[int(label)].append(i)
    
    print(f"Found {len(class_to_indices)} classes in training dataset")
    return class_to_indices

def load_train_samples_batch(train_dataset, indices, device):
    """Load a batch of training samples efficiently"""
    return torch.stack([train_dataset[i][0] for i in indices]).to(device)

def evaluate_gi_on_adversarials(model, train_dataset, adv_dir, dataset_name, device, max_train=100):
    # Pre-organize training dataset by class
    class_to_indices = organize_train_dataset_by_class(train_dataset)
    
    # Pre-load training samples for each class to avoid repeated loading
    print("Pre-loading training samples for each class...")
    class_samples_cache = {}
    for class_id, indices in class_to_indices.items():
        if len(indices) > 0:
            # Randomly select a subset of samples for this class
            selected_indices = np.random.choice(
                indices, 
                size=min(max_train, len(indices)), 
                replace=False
            )
            # Load and cache these samples
            class_samples_cache[class_id] = load_train_samples_batch(train_dataset, selected_indices, device)
            print(f"Cached {len(selected_indices)} samples for class {class_id}")
    
    correct_records, wrong_records = [], []
    
    # Create a list of all adversarial image paths to process
    adv_image_paths = []
    for class_folder in os.listdir(adv_dir):
        folder_path = os.path.join(adv_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                adv_image_paths.append((
                    os.path.join(folder_path, file),
                    os.path.join(folder_path, file.replace(".png", ".json"))
                ))
    
    # Process adversarial examples
    for img_path, meta_path in tqdm(adv_image_paths, desc="Processing adversarial examples"):
        with open(meta_path) as f:
            meta = json.load(f)

        x = get_image_tensor(img_path, dataset_name).to(device)
        y_true = meta["original"]
        sample_id = meta["index"]

        with torch.no_grad():
            y_pred = model(x).argmax(1).item()
            if dataset_name == "imagenette":
                y_pred = IMAGENET_TO_IMAGENETTE.get(y_pred, -1)

        if y_pred not in class_samples_cache or y_pred == -1:
            continue
        # print(y_pred, y_true)
        # Use the pre-cached training samples
        train_images = class_samples_cache[y_pred]
        # print(f"Processing sample {sample_id} with true label {y_true} and predicted label {y_pred}. Train images: {len(train_images)}")
        # Calculate GIs
        gis, _ = compute_GIs_for_sample(
            model=model,
            sample=x.squeeze(0),
            class_samples=train_images,
            correct_class=y_pred,
            step_size=0.01,
            max_eps=3.0,
            device=device,
            normalized=False,
            batch_size=6000  # Adjust based on GPU memory
        )
        # print(f"GI calculation completed for sample {sample_id}")
        # Store only summary statistics to reduce memory usage
        percent_above_1 = 100 * sum(g > 1 for g in gis) / len(gis)
        mean_gi = np.mean(gis)
        median_gi = np.median(gis)
        max_gi = np.max(gis)
        min_gi = np.min(gis)
        
        row = {
            "sample_id": sample_id,
            "true_label": y_true,
            "predicted_label": y_pred,
            "percent_gi_above_1": percent_above_1,
            "mean_gi": mean_gi,
            "median_gi": median_gi,
            "max_gi": max_gi,
            "min_gi": min_gi,
            # Store only the first 20 individual GIs to keep file size manageable
            **{f"gi_{i}": v for i, v in enumerate(gis[:20])}
        }

        # Save to appropriate record list
        if y_true == y_pred:
            correct_records.append(row)
        else:
            wrong_records.append(row)
            
        # Save results every 100 samples to prevent memory issues
        if (len(correct_records) + len(wrong_records)) % 100 == 0:
            print(f"Processed {len(correct_records) + len(wrong_records)} samples. "
                  f"Saving intermediate results...")
            
            pd.DataFrame(correct_records).to_csv(
                os.path.join(adv_dir, "correct_gi_intermediate.csv"), 
                index=False
            )
            pd.DataFrame(wrong_records).to_csv(
                os.path.join(adv_dir, "wrong_gi_intermediate.csv"), 
                index=False
            )

    return correct_records, wrong_records

# --------- Entry Point ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'imagenette'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--adv_dir', type=str, required=True)
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from src.runtime import seed_all, device_for
    seed_all(args.seed)
    device = str(device_for(args.device))
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = load_model(args.model, args.dataset, device)
    print("Model loaded successfully!")
    
    print("Setting up dataset...")
    transform = get_transform(args.dataset)
    train_dataset = load_train_dataset(args.dataset, args.data_path, transform)
    print(f"Dataset loaded with {len(train_dataset)} samples")

    print("Starting GI evaluation on adversarial examples...")
    correct_records, wrong_records = evaluate_gi_on_adversarials(
        model=model,
        train_dataset=train_dataset,
        adv_dir=args.adv_dir,
        dataset_name=args.dataset,
        device=device,
        max_train=args.max_train
    )

    pd.DataFrame(correct_records).to_csv(os.path.join(args.adv_dir, "correct_gi.csv"), index=False)
    pd.DataFrame(wrong_records).to_csv(os.path.join(args.adv_dir, "wrong_gi.csv"), index=False)

    print(f"GI stats saved. Correct: {len(correct_records)}, Wrong: {len(wrong_records)}")