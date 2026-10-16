"""Auxiliary losses for expanding or restricting directional prediction stability.

The perturbation and loss definitions match the MNIST MLP intervention."""
import torch
import torch.nn.functional as F


def margin(logits, labels):
    """f^y(x) - max_{i != y} f^i(x), batched. Matches Definition 3 in the paper."""
    correct_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    max_other = masked.max(dim=1).values
    return correct_logit - max_other


def random_unit_directions(x):
    """Isotropic Gaussian direction per sample, L2-normalized (Fig. 1 right-panel convention)."""
    u = torch.randn_like(x)
    flat = u.view(u.size(0), -1)
    norms = flat.norm(dim=1).clamp_min(1e-8)
    flat = flat / norms.view(-1, 1)
    return flat.view_as(x)


def perturbation_step(x, eps_frac):
    """eps(x) = eps_frac * ||x||_2, per-sample self-calibrating step size."""
    flat = x.view(x.size(0), -1)
    return eps_frac * flat.norm(dim=1)


def intervention_loss(model, x, y, arm, lam, tau, eps_frac):
    """
    Returns (total_loss, ce_loss, hinge_loss, perturbed_margin_mean) for one batch.

    arm: 'control' | 'expand' | 'restrict'
    """
    logits = model(x)
    ce = F.cross_entropy(logits, y)

    if arm == "control" or lam == 0.0:
        zero = torch.zeros((), device=x.device)
        return ce, ce.detach(), zero, torch.tensor(float("nan"))

    u = random_unit_directions(x)
    eps = perturbation_step(x, eps_frac).view(-1, *([1] * (x.dim() - 1)))
    x_pert = x + eps * u

    logits_pert = model(x_pert)
    m_pert = margin(logits_pert, y)

    if arm == "expand":
        # reward margin(x + eps*u) >= +tau: prediction stays correct & confident nearby
        hinge = F.relu(tau - m_pert).mean()
    elif arm == "restrict":
        # reward margin(x + eps*u) <= -tau: prediction becomes confidently wrong nearby
        hinge = F.relu(m_pert + tau).mean()
    else:
        raise ValueError(f"unknown arm: {arm}")

    total = ce + lam * hinge
    return total, ce.detach(), hinge.detach(), m_pert.mean().detach()
