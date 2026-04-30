# Unified task wrapper for PMTL-style models.
# Replaces RegressionTrain (model_lenet.py) and RegressionTrainResNet (model_resnet.py).

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskWrapper(nn.Module):
    """Wraps any backbone that returns (B, n_tasks, n_classes) and computes per-task CE losses."""

    def __init__(self, backbone: nn.Module, n_tasks: int, init_weight=None) -> None:
        super().__init__()
        self.model = backbone
        self.n_tasks = n_tasks
        if init_weight is None:
            w = torch.ones(n_tasks) / n_tasks
        elif isinstance(init_weight, np.ndarray):
            w = torch.from_numpy(init_weight).float()
        else:
            w = torch.tensor(init_weight).float()
        self.weights = nn.Parameter(w)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        ys = self.model(x)  # (B, n_tasks, n_classes)
        losses = [self.ce_loss(ys[:, i], ts[:, i]) for i in range(self.n_tasks)]
        return torch.stack(losses)  # (n_tasks,)
