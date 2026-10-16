import torch 
import numpy as np
from tqdm import tqdm

def compute_GI(model, x1, x2, direction_normalized, correct_class, step_size=0.0001, max_eps=3.0, device="cpu"):
    """
    Compute GI with option for normalized or unnormalized direction.
    Vectorized implementation for batch processing.
    """
    model.eval()
    with torch.no_grad():
        # Ensure inputs are batched
        if x1.dim() == 3:  # If input is [C, H, W]
            x1 = x1.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
            
        batch_size = x1.size(0)
        
        # Move tensors to device and compute directions
        x1, x2 = x1.to(device), x2.to(device)
        directions = x2 - x1  # Shape: [batch_size, C, H, W]
        
        if direction_normalized:
            norms = torch.norm(directions.view(batch_size, -1), dim=1)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            directions = directions / norms.view(batch_size, 1, 1, 1)

        # Initialize epsilon values
        pos_eps = torch.full((batch_size,), max_eps, device=device)
        neg_eps = torch.full((batch_size,), max_eps, device=device)
        
        # Create steps tensor
        steps = torch.arange(0, max_eps + step_size, step_size, device=device)
        
        count = 0
        # Positive direction search
        for eps in steps:
            # Compute points for all samples in batch
            points = x1 + eps * directions
            # Get predictions for all points
            outputs = model(points)
            preds = outputs.argmax(1)
            changed_mask = (preds != correct_class) & (pos_eps == max_eps)
            # count += sum(changed_mask)
            pos_eps[changed_mask] = eps
            
        # Negative direction search
        for eps in steps:
            # Compute points for all samples in batch
            points = x1 - eps * directions
            # Get predictions for all points
            outputs = model(points)
            preds = outputs.argmax(1)
            # Find which predictions changed from correct_class
            changed_mask = (preds != correct_class) & (neg_eps == max_eps)

            neg_eps[changed_mask] = eps

        return pos_eps.cpu().numpy(), neg_eps.cpu().numpy()


def compute_GIs_for_sample(model, sample, class_samples, correct_class, normalized=False, step_size=0.001, max_eps=3.0, device="cpu", batch_size=6000):
    """
    Compute GIs for a sample against all class samples using batch processing.
    """
    # Move sample to device and ensure it's a tensor
    sample = torch.tensor(sample, device=device) if not isinstance(sample, torch.Tensor) else sample.to(device)
    
    # Convert class_samples to tensor if it's a list
    if isinstance(class_samples, list):
        class_samples = torch.stack([torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in class_samples])
    class_samples = class_samples.to(device)
    
    # Process in batches
    batch_size = batch_size  # Adjust based on your GPU memory
    num_samples = len(class_samples)
    pos_gis = []
    neg_gis = []
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_samples = class_samples[i:batch_end]
        
        # Expand sample to match batch size
        batch_target = sample.unsqueeze(0).expand(len(batch_samples), -1, -1, -1)
        
        # Compute GIs for batch
        pos_batch, neg_batch = compute_GI(
            model, batch_samples, batch_target, 
            direction_normalized=normalized,
            correct_class=correct_class, 
            step_size=step_size, 
            max_eps=max_eps, 
            device=device
        )
        
        pos_gis.extend(pos_batch)
        neg_gis.extend(neg_batch)
    
    return np.array(pos_gis), np.array(neg_gis)
