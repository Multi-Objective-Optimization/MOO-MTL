# MTL evaluation metrics.
# Merged from PMTL (accuracy computation in train.py) and CPMTL (pareto/metrics.py).

from typing import Iterable
import numpy as np
import torch
from torch import Tensor


def topk_accuracies(output: Tensor, label: Tensor, ks: Iterable[int] = (1,)):
    """Compute top-k accuracies for a single task."""
    assert output.dim() == 2 and label.dim() == 1
    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    label = label.unsqueeze(1).expand_as(pred)
    correct = pred.eq(label).float()
    return [correct[:, :k].sum(1).mean().item() for k in ks]


def topk_accuracy(output: Tensor, label: Tensor, k: int = 1) -> float:
    return topk_accuracies(output, label, (k,))[0]


@torch.no_grad()
def compute_accuracy(model_backbone, loader, n_tasks: int, device) -> list:
    """Compute per-task top-1 accuracy. For PMTL-style (B, n_tasks, n_classes) outputs."""
    correct = [0] * n_tasks
    total = 0
    model_backbone.eval()
    for X, ts in loader:
        X, ts = X.to(device), ts.to(device)
        out = model_backbone(X)  # (B, n_tasks, n_classes)
        for i in range(n_tasks):
            pred = out[:, i].argmax(dim=1)
            correct[i] += pred.eq(ts[:, i]).sum().item()
        total += X.size(0)
    return [c / total for c in correct]


@torch.no_grad()
def evaluate_cpmtl(network, dataloader, device, closures) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate CPMTL-style network (returns tuple of logits per task)."""
    num_samples = 0
    losses = np.zeros(len(closures))
    top1s = np.zeros(len(closures))
    network.eval()
    for images, labels in dataloader:
        batch_size = len(images)
        num_samples += batch_size
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        losses_batch = [c(network, logits, labels).item() for c in closures]
        losses += batch_size * np.array(losses_batch)
        for i, logit in enumerate(logits):
            top1s[i] += batch_size * topk_accuracy(logit, labels[:, i])
    losses /= num_samples
    top1s /= num_samples
    return losses, top1s
