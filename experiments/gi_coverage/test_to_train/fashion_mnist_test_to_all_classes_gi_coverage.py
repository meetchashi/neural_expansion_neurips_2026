
import os
os.environ["DISPLAY"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib
matplotlib.use('Agg')

import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, Subset
from src.analysis.gi_calculation import compute_GIs_for_sample
import yaml
import importlib

def get_image_by_index(dataset, index):
    return dataset[index][0]

def load_model_from_config(config_path, checkpoint_path, device):
    """Load model based on configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_config = config['model']
    module_path = model_config.get('module', 'src.models')
    model_file = model_config.get('file', model_config['name'].lower())
    
    try:
        module = importlib.import_module(f"{module_path}.{model_file}")
        model_class = getattr(module, model_config['name'])
        model = model_class(**model_config.get('args', {})).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, config
    except (ImportError, AttributeError) as e:
        print(f"Error loading model: {e}")
        raise

def cluster_train_by_labels(train_dataset, total_class=10):
    clusters = {i: [] for i in range(total_class)}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
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

def run_experiment(config_path, checkpoint_path, data_path, output_path, k=None):
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, config = load_model_from_config(config_path, checkpoint_path, device)
    model_type = config['model']['name'].lower()
    dataset_config = config['dataset']
    total_classes = dataset_config.get('num_classes', 10)
    
    transform_config = dataset_config.get('transform', {}
    )
    transform_list = [transforms.ToTensor()]
    if 'normalize' in transform_config:
        transform_list.append(transforms.Normalize(transform_config['normalize']['mean'], transform_config['normalize']['std']))
    transform = transforms.Compose(transform_list)

    full_train = FashionMNIST(root=data_path, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=data_path, train=False, transform=transform, download=True)
    val_size = int(dataset_config.get('val_split', 0.2) * len(full_train))
    train_dataset, _ = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(config.get('seed', 42)))

    train_indices_by_class = cluster_train_by_labels(train_dataset, total_class=total_classes)

    # -----------------------------------------------------------------------
    # Training-set inference → misclassified sample IDs per class
    # -----------------------------------------------------------------------
    print("Running model inference on training split …")
    misclassified_train_ids = {c: [] for c in range(total_classes)}
    batch_size_inf = 512
    all_train_indices = list(train_dataset.indices)
    with torch.no_grad():
        for i in range(0, len(all_train_indices), batch_size_inf):
            batch_idx = all_train_indices[i:i + batch_size_inf]
            imgs   = torch.stack([full_train[j][0] for j in batch_idx]).to(device)
            labels = [full_train[j][1] for j in batch_idx]
            preds  = model(imgs).argmax(dim=1).cpu().tolist()
            for j, (pred, true) in enumerate(zip(preds, labels)):
                if pred != true:
                    misclassified_train_ids[true].append(int(batch_idx[j]))

    # -----------------------------------------------------------------------
    # Test-sample GI loop
    # -----------------------------------------------------------------------
    test_indices = []
    if k is not None:
        for i in range(total_classes):
            class_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label == i]
            test_indices.extend(np.random.choice(class_indices, size=k, replace=False))
    else:
        test_indices = list(range(len(test_dataset)))

    results = []

    for idx in tqdm(test_indices, desc='Processing test samples'):
        test_img, true_label = test_dataset[idx]
        test_img = test_img.to(device).unsqueeze(0)
        pred = torch.argmax(model(test_img)).item()

        sample_result = {
            'sample_id': idx,
            'true_label': true_label,
            'predicted_class': pred
        }

        for target_class_id in range(total_classes):
            train_indices_for_target_class = train_indices_by_class.get(target_class_id, [])
            
            if train_indices_for_target_class:
                train_imgs = torch.stack([get_image_by_index(train_dataset, i) for i in train_indices_for_target_class]).to(device)
                gis = calculate_gis_from_test_to_train(model, test_img.squeeze(0), train_imgs, target_class_id, device)
                if gis.size > 0:
                    percent_gi_above_1 = 100 * np.sum(gis > 1.0) / len(gis)
                    # Save raw GI per training sample: column name encodes class + original dataset index
                    for gi_val, train_ds_idx in zip(gis, train_indices_for_target_class):
                        # train_ds_idx is the index into train_dataset (0-based within split)
                        # we use full_train index via train_dataset.indices for a stable global ID
                        global_train_idx = train_dataset.indices[train_ds_idx]
                        sample_result[f'gi_c{target_class_id}_{global_train_idx}'] = float(gi_val)
                else:
                    percent_gi_above_1 = 0
            else:
                percent_gi_above_1 = 0

            sample_result[f'coverage_class_{target_class_id}'] = percent_gi_above_1
        
        results.append(sample_result)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    base_output_path = os.path.join(output_path, f'GI_all_classes_experiment_outputs_{model_type}')
    os.makedirs(base_output_path, exist_ok=True)

    # CSV: existing coverage columns + new per-training-sample GI columns
    out_csv = os.path.join(base_output_path, 'test_to_all_classes_gi_coverage.csv')
    df = pd.DataFrame(results)
    # Put summary coverage columns first for readability
    summary_cols = ['sample_id', 'true_label', 'predicted_class'] + \
                   [f'coverage_class_{c}' for c in range(total_classes)]
    gi_cols = [c for c in df.columns if c not in summary_cols]
    df = df[summary_cols + gi_cols]
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV ({len(df)} rows, {len(df.columns)} columns) → {out_csv}")

    # JSON: misclassified training sample IDs per class
    import json
    out_json = os.path.join(base_output_path, 'misclassified_train_ids.json')
    with open(out_json, 'w') as f:
        json.dump({str(c): ids for c, ids in misclassified_train_ids.items()}, f, indent=2)
    total_mis = sum(len(v) for v in misclassified_train_ids.values())
    print(f"Saved misclassified training IDs ({total_mis} total across all classes) → {out_json}")
    for c in range(total_classes):
        n = len(misclassified_train_ids[c])
        total_c = len(train_indices_by_class[c])
        print(f"  Class {c}: {n} / {total_c} training samples misclassified ({100*n/total_c:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MNIST dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--k', type=int, default=10, help='Randomly pick k test samples PER CLASS (default: 10)')
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k
    )

'''
# Example Command:

python3 experiments/gi_coverage/test_to_train/fashion_mnist_test_to_all_classes_gi_coverage.py \
  --config ./configs/training/fashion_mnist_cnn.yaml \
  --checkpoint ./checkpoints/fashion_mnist_cnn/best_model_fashionmnist.pth \
  --data_path ./data/FashionMNIST-Original \
  --output_path ./experiments/results/temp_all_classes/fashion_mnist_cnn \
  --k 5
'''
