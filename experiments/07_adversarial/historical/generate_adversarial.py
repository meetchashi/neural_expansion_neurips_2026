import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import timm
import torch.nn.functional as F

# Label mapping for ImageNette
IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701
}
IMAGENET_TO_IMAGENETTE = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}

# --------- Attack Implementations ----------

def iterative_fgsm(model, x, y, eps=0.03, alpha=0.005, iters=10):
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(iters):
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv.data = x_adv.data + alpha * x_adv.grad.sign()
        x_adv.data = torch.clamp(x_adv.data, x - eps, x + eps).clamp(0, 1)
        x_adv.grad.zero_()
    return x_adv.detach()

def pgd(model, x, y, eps=0.03, alpha=0.005, iters=100):
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach().requires_grad_(True)
    for _ in range(iters):
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv.data = x_adv.data + alpha * x_adv.grad.sign()
        x_adv.data = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv.data = x_adv.data.clamp(0, 1)
        x_adv.grad.zero_()
    return x_adv.detach()

def cw_l2(model, x, y_target, c=1e-2, lr=1e-2, iters=1000):
    x_var = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_var], lr=lr)
    for _ in range(iters):
        logits = model(x_var)
        real = F.cross_entropy(logits, y_target)
        l2 = F.mse_loss(x_var, x)
        loss = l2 + c * real
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x_var.data = x_var.data.clamp(0, 1)
    return x_var.detach()

# --------- Utilities ----------

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
    elif dataset_name == "imagenette":
        if model_arg == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_arg == "VGG16":
            model = models.vgg16(pretrained=True)
        elif model_arg == "ViT-Base":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
        elif model_arg == "DeiT-Small":
            model = timm.create_model("deit_small_patch16_224", pretrained=True)
    return model.to(device).eval()

def load_dataset(dataset_name, data_path):
    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
    elif dataset_name == "imagenette":
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
    return dataset

# --------- Main Generator ----------

def generate_adversarials(model, dataset, dataset_name, save_root, attack, k, device):
    # For ImageNette, we already know there are 10 classes
    num_classes = 10
    os.makedirs(save_root, exist_ok=True)

    imagenet_to_imagenette = IMAGENET_TO_IMAGENETTE if dataset_name == "imagenette" else None
    imagenette_to_imagenet = IMAGENETTE_TO_IMAGENET if dataset_name == "imagenette" else None

    all_clean_preds, all_adv_preds, all_labels = [], [], []
    
    # Pre-organize indices by class - MUCH faster approach
    print("Organizing dataset indices by class...")
    indices_by_class = {}
    
    # For ImageFolder datasets like ImageNette, use the faster .imgs attribute
    if dataset_name == "imagenette" and hasattr(dataset, 'imgs'):
        for i, (_, label) in enumerate(dataset.imgs):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(i)
        print("Used fast path for ImageFolder dataset")
    else:
        # Fallback to using DataLoader for faster processing
        label_loader = DataLoader(
            dataset, 
            batch_size=256,
            num_workers=4,
            shuffle=False
        )
        
        idx = 0
        for _, batch_labels in tqdm(label_loader):
            for label in batch_labels:
                label = label.item()
                if label not in indices_by_class:
                    indices_by_class[label] = []
                indices_by_class[label].append(idx)
                idx += 1
    
    for class_id in tqdm(range(num_classes)):
        print(f"Processing class {class_id}...")
        # Get pre-computed indices for this class
        indices = indices_by_class.get(class_id, [])
        print(f"Found {len(indices)} samples for class {class_id}")
        
        np.random.shuffle(indices)
        selected = 0
        clean_preds, adv_preds, labels = [], [], []

        for i in tqdm(indices[:k], desc=f"Generating adversarials for class {class_id}"):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = torch.tensor([y], device=device)

            with torch.no_grad():
                pred = model(x)
                pred_label = pred.argmax(1).item()
                if dataset_name == "imagenette":
                    pred_label = imagenet_to_imagenette.get(pred_label, -1)
                if pred_label != y.item():
                    continue

            # MODIFIED: Choose target from available ImageNette classes instead of random
            if dataset_name == "imagenette":
                # Get all classes except the current one
                available_targets = [c for c in range(num_classes) if c != y.item()]
                target_idx = torch.randint(0, len(available_targets), (1,), device=device)
                target = torch.tensor([available_targets[target_idx]], device=device)
                
                # Convert to ImageNet label for the model if needed
                if attack == "cw":
                    target_imagenet = torch.tensor([imagenette_to_imagenet[target.item()]], device=device)
                    x_adv = cw_l2(model, x, target_imagenet)
                else:
                    # For PGD and IFGSM, we need the original label, not the target
                    if attack == "ifgsm":
                        x_adv = iterative_fgsm(model, x, y)
                    elif attack == "pgd":
                        # Reduce iterations for faster processing
                        x_adv = pgd(model, x, y, iters=30)  # Reduced from 100 to 30
            else:
                # Original behavior for non-imagenette datasets
                target = (y + torch.randint(1, num_classes, (1,), device=device)) % num_classes
                
                if attack == "ifgsm":
                    x_adv = iterative_fgsm(model, x, y)
                elif attack == "pgd":
                    x_adv = pgd(model, x, y, iters=30)  # Reduced iterations
                elif attack == "cw":
                    x_adv = cw_l2(model, x, target)
                else:
                    raise ValueError("Unknown attack")

            save_dir = os.path.join(save_root, f"class_{class_id}")
            os.makedirs(save_dir, exist_ok=True)
            save_image(x_adv, os.path.join(save_dir, f"sample_{i}.png"))
            meta = {"original": y.item(), "target": int(target.item()), "index": int(i)}
            with open(os.path.join(save_dir, f"sample_{i}.json"), "w") as f:
                json.dump(meta, f)

            with torch.no_grad():
                clean_pred = model(x).argmax(1).item()
                adv_pred = model(x_adv).argmax(1).item()
                if dataset_name == "imagenette":
                    clean_pred = imagenet_to_imagenette.get(clean_pred, -1)
                    adv_pred = imagenet_to_imagenette.get(adv_pred, -1)

            clean_preds.append(clean_pred)
            adv_preds.append(adv_pred)
            labels.append(y.item())

            all_clean_preds.append(clean_pred)
            all_adv_preds.append(adv_pred)
            all_labels.append(y.item())

            selected += 1
            if selected >= k:
                break


        # Save per-class report
        if labels:
            acc_clean = accuracy_score(labels, clean_preds)
            acc_adv = accuracy_score(labels, adv_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, adv_preds, average='macro', zero_division=0)
            report = {
                "clean_accuracy": acc_clean,
                "adv_accuracy": acc_adv,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "num_samples": len(labels)
            }
            with open(os.path.join(save_dir, "adv_report.json"), "w") as f:
                json.dump(report, f, indent=2)

    # Save overall report
    if all_labels:
        acc_clean = accuracy_score(all_labels, all_clean_preds)
        acc_adv = accuracy_score(all_labels, all_adv_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_adv_preds, average='macro', zero_division=0)
        report = {
            "clean_accuracy": acc_clean,
            "adv_accuracy": acc_adv,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "num_samples": len(all_labels)
        }
        with open(os.path.join(save_root, "adv_overall_report.json"), "w") as f:
            json.dump(report, f, indent=2)

# --------- CLI ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name or .pth checkpoint")
    parser.add_argument('--dataset', type=str, choices=["mnist", "cifar10", "imagenette"], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--attack', type=str, choices=["ifgsm", "pgd", "cw"], default="pgd")
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--k', type=int, default=5, help="Samples per class")
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

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")

    print("Starting adversarial generation...")
    generate_adversarials(model, dataset, args.dataset, args.output, args.attack, args.k, device)
