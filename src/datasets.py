"""Deterministic dataset recipes; no images or labels are distributed."""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

LT_COUNTS = {
    'lt1': [4717, 2866, 2370, 2015, 1806, 1590, 1346, 1034, 886, 730],
    'lt2': [4716, 2504, 2026, 1464, 1096, 761, 383, 218, 93, 60],
}

class Relabeled(Dataset):
    def __init__(self, dataset, classes, seed):
        self.dataset = dataset
        self.targets = np.random.RandomState(seed).randint(classes, size=len(dataset))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx][0], int(self.targets[idx])

def load_custom_datasets(config):
    norm = config.get('transform', {}).get('normalize', {'mean': [0.5], 'std': [0.5]})
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
    name = config['name'].lower().replace('_', '').replace('-', '')
    cls = {'mnist': datasets.MNIST, 'fashionmnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10}[name]
    train, test = [cls(config['data_dir'], train=is_train, download=True, transform=transform) for is_train in (True, False)]
    variant = config.get('variant', 'original')
    if variant.startswith('random'):
        classes = int(variant.removeprefix('random'))
        seed = config.get('variant_seed', 42)
        train = Relabeled(train, classes, seed)
        if classes == 100: test = Relabeled(test, classes, seed + 1)
    return train, test

def apply_training_variant(train, config):
    variant = config.get('variant', 'original')
    if variant not in LT_COUNTS: return train
    rng = np.random.RandomState(config.get('variant_seed', 42))
    labels = np.array([train[i][1] for i in range(len(train))])
    chosen = []
    for c, count in enumerate(LT_COUNTS[variant]):
        pool = np.flatnonzero(labels == c)
        if len(pool) < count: raise ValueError(f'Class {c}: need {count} samples, found {len(pool)}')
        chosen.extend(rng.choice(pool, count, replace=False).tolist())
    # Flatten the subset so saved indices refer to the underlying dataset.
    return Subset(train.dataset, [train.indices[i] for i in chosen])

def training_split(config):
    train, test = load_custom_datasets(config['dataset'])
    nval = int(config['dataset'].get('val_split', 0.2) * len(train))
    train, val = random_split(train, [len(train)-nval, nval], generator=torch.Generator().manual_seed(config.get('seed', 42)))
    return apply_training_variant(train, config['dataset']), val, test
