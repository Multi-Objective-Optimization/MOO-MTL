# Multi-task LeNet backbone.
# Merged from PMTL (model_lenet.py) and CPMTL (multi_lenet.py).
# PMTL version: larger conv filters (9x9, 5x5), 36x36 input → 5x5x20 features → 50-dim.
# CPMTL version: standard conv filters (5x5, 5x5), 28x28 input → 4x4x20 features → 50-dim.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor


class LeNetPMTL(nn.Module):
    """Multi-task LeNet from PMTL. Input: (B, 1, 36, 36)."""

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)
        for i in range(n_tasks):
            setattr(self, f"task_{i}", nn.Linear(50, 10))

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))
        outs = [getattr(self, f"task_{i}")(x) for i in range(self.n_tasks)]
        return torch.stack(outs, dim=1)  # (B, n_tasks, 10)


class LeNetCPMTL(nn.Module):
    """Multi-task LeNet from CPMTL. Input: (B, 1, 28, 28).

    Returns tuple (logit_task0, logit_task1) for closure-based loss.
    Supports shared_parameters() for HVP computation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (5, 5))
        self.conv2 = nn.Conv2d(10, 20, (5, 5))
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc3_1 = nn.Linear(50, 10)
        self.fc3_2 = nn.Linear(50, 10)

    def shared_parameters(self) -> List[Tensor]:
        return [p for n, p in self.named_parameters() if not n.startswith("fc3")]

    def forward(self, x: Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc3_1(x), self.fc3_2(x)
