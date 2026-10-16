import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import random
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import click
import logging
import yaml
import importlib
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model(model_config):
    """Dynamically import and instantiate the model based on config."""
    model_name = model_config['name']
    model_args = model_config.get('args', {})
    
    # Import the module containing the model
    module_path = model_config.get('module', 'src.models')
    
    # Use the 'file' field from config if available, otherwise fallback to lowercased model name
    model_file = model_config.get('file', model_name.lower())
    
    # Import the module
    module = importlib.import_module(f"{module_path}.{model_file}")
    
    # Get the model class and instantiate it
    model_class = getattr(module, model_name)
    return model_class(**model_args)

def calculate_classwise_metrics(outputs, labels, criterion):
    """Calculate metrics for each class."""
    _, predicted = torch.max(outputs, 1)
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    class_loss = torch.zeros(10)
    
    losses = criterion(outputs, labels)
    
    for i in range(len(labels)):
        label = labels[i]
        class_total[label] += 1
        class_loss[label] += losses[i]
        if predicted[i] == label:
            class_correct[label] += 1
    
    return class_correct, class_total, class_loss

def update_metrics_csv(checkpoint_name, total_acc, class_acc, filename):
    """Update CSV file with metrics."""
    row = {'checkpoint': checkpoint_name, 'total_accuracy': float(total_acc)}
    for i, acc in enumerate(class_acc):
        row[f'class_{i}_accuracy'] = float(acc)
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(filename, index=False)

def evaluate(model, loader, criterion, device, num_classes=10):
    """Evaluate model on a given data loader."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    avg_loss = running_loss / len(loader)
    total_acc = 100 * correct / total
    class_acc = [100 * (c.cpu() / t.cpu() if t > 0 else 0) for c, t in zip(class_correct, class_total)]
    
    return total_acc, class_acc, avg_loss

def get_optimizer(optimizer_config, model_parameters):
    """Get optimizer based on config."""
    optimizer_name = optimizer_config['name']
    optimizer_args = optimizer_config.get('args', {})
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_parameters, **optimizer_args)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_parameters, **optimizer_args)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_parameters, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(scheduler_config, optimizer):
    """Get learning rate scheduler based on config."""
    if scheduler_config is None:
        return None
        
    scheduler_name = scheduler_config['name']
    scheduler_args = scheduler_config.get('args', {})
    
    if scheduler_name.lower() == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
    elif scheduler_name.lower() == 'multisteplr':
        return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
    elif scheduler_name.lower() == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
    elif scheduler_name.lower() == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def get_dataset(dataset_config):
    from src.datasets import load_custom_datasets
    return load_custom_datasets(dataset_config)


def save_and_evaluate_checkpoint(model, epoch, batch_idx, train_loader, val_loader, test_loader, 
                              criterion, device, checkpoint_dir, metrics_files, metrics, 
                              best_val_accuracy, best_model_path, num_classes):
    """Save checkpoint and evaluate the model."""
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_epoch{epoch}_batch{batch_idx}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Evaluate on train, validation, and test sets
    model.eval()
    train_acc, train_class_acc, train_loss = evaluate(model, train_loader, criterion, device, num_classes)
    val_acc, val_class_acc, val_loss = evaluate(model, val_loader, criterion, device, num_classes)
    test_acc, test_class_acc, test_loss = evaluate(model, test_loader, criterion, device, num_classes)
    
    # Save current metrics
    checkpoint_name = f"epoch{epoch}_batch{batch_idx}"
    update_metrics_csv(checkpoint_name, train_acc, train_class_acc, metrics_files['train'])
    update_metrics_csv(checkpoint_name, val_acc, val_class_acc, metrics_files['val'])
    update_metrics_csv(checkpoint_name, test_acc, test_class_acc, metrics_files['test'])
    
    # Save best model based on validation accuracy
    best_model_updated = False
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        best_model_updated = True
    
    # Update metrics
    metrics["epoch"].append(epoch + batch_idx/len(train_loader))
    metrics["train_loss"].append(train_loss)
    metrics["train_accuracy"].append(train_acc)
    metrics["val_loss"].append(val_loss)
    metrics["val_accuracy"].append(val_acc)
    metrics["test_loss"].append(test_loss)
    metrics["test_accuracy"].append(test_acc)
    metrics["train_class_acc"].append(train_class_acc)
    metrics["val_class_acc"].append(val_class_acc)
    metrics["test_class_acc"].append(test_class_acc)
    
    model.train()  # Return to training mode
    
    return best_val_accuracy, best_model_updated

def train_model(config_path):
    """Train model based on configuration file."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup directories and logging
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_filename = os.path.join(checkpoint_dir, 'training.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Set random seed
    set_seed(config.get('seed', 42))

    # Determine device
    device_name = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    
    # Load dataset
    train_dataset, test_dataset = get_dataset(config['dataset'])
    
    # Create train/validation split
    val_size = int(config['dataset'].get('val_split', 0.2) * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.get('seed', 42)))
    
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    from src.datasets import apply_training_variant
    train_dataset = apply_training_variant(train_dataset, config['dataset'])
    np.savez(os.path.join(checkpoint_dir, 'split_indices.npz'), train=train_dataset.indices, val=val_dataset.indices)

    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup metrics files
    dataset_name = config['dataset']['name'].lower()
    metrics_files = {
        'train': os.path.join(checkpoint_dir, f'{dataset_name}_train_checkpoints.csv'),
        'val': os.path.join(checkpoint_dir, f'{dataset_name}_val_checkpoints.csv'),
        'test': os.path.join(checkpoint_dir, f'{dataset_name}_test_checkpoints.csv')
    }
    
    # Initialize model
    model = get_model(config['model']).to(device)
    logging.info(f"Model: {model}")
    
    # Initialize criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config['optimizer'], model.parameters())
    scheduler = get_scheduler(config.get('scheduler'), optimizer)
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    save_every = config['training'].get('save_every', len(train_loader)-1)
    num_classes = config['dataset'].get('num_classes', 10)
    
    # Initialize metrics
    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "train_class_acc": [],
        "val_class_acc": [],
        "test_class_acc": []
    }
    
    # Initialize best validation tracking
    best_val_accuracy = 0.0
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{dataset_name}.pth")
    
    # Save the initial random model (epoch 0, batch 0)
    best_val_accuracy, _ = save_and_evaluate_checkpoint(
        model, 0, 0, train_loader, val_loader, test_loader,
        criterion, device, checkpoint_dir, metrics_files, metrics,
        best_val_accuracy, best_model_path, num_classes
    )
    
    # Main training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # Perform training step
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Save checkpoint and evaluate at specified intervals or at the end of the epoch
            if batch_idx % save_every == 0 and batch_idx > 0:  # Skip batch 0, as we already saved it
                best_val_accuracy, _ = save_and_evaluate_checkpoint(
                    model, epoch, batch_idx, train_loader, val_loader, test_loader,
                    criterion, device, checkpoint_dir, metrics_files, metrics,
                    best_val_accuracy, best_model_path, num_classes
                )
        
        # Always save at the end of each epoch
        if batch_idx == 0 or batch_idx % save_every != 0:  # Avoid duplicate saves if save_every coincides with last batch
            best_val_accuracy, _ = save_and_evaluate_checkpoint(
                model, epoch, len(train_loader)-1, train_loader, val_loader, test_loader,
                criterion, device, checkpoint_dir, metrics_files, metrics,
                best_val_accuracy, best_model_path, num_classes
            )
        
        # Step the scheduler if it exists
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["val_loss"][-1])
            else:
                scheduler.step()
        
        logging.info(f"Epoch {epoch}: Train Loss: {metrics['train_loss'][-1]:.4f}, Val Loss: {metrics['val_loss'][-1]:.4f}")
    
    # Save final plots
    save_training_plots(metrics, checkpoint_dir, dataset_name, num_classes)

def save_training_plots(metrics, save_dir, dataset_name, num_classes=10):
    """Save training plots."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["epoch"], metrics["val_loss"], label="Val Loss")
    plt.plot(metrics["epoch"], metrics["test_loss"], label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics["epoch"], metrics["train_accuracy"], label="Train Acc")
    plt.plot(metrics["epoch"], metrics["val_accuracy"], label="Val Acc")
    plt.plot(metrics["epoch"], metrics["test_accuracy"], label="Test Acc")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_overall_metrics.png'))
    plt.close()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        plt.plot(metrics["epoch"], [acc[i] for acc in metrics["train_class_acc"]], 
                label=f"Class {i}")
    plt.title("Classwise Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        plt.plot(metrics["epoch"], [acc[i] for acc in metrics["val_class_acc"]], 
                label=f"Class {i}")
    plt.title("Classwise Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_classwise_metrics.png'))
    plt.close()

@click.command()
@click.option('--config', required=True, help='Path to the configuration YAML file')
def main(config):
    train_model(config)

if __name__ == "__main__":
    main()
