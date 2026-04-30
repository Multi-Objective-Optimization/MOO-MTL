# Multi-task ResNet18 backbone for PMTL.
# Input: (B, 1, 36, 36) — grayscale.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


class ResNet18MTL(nn.Module):
    """ResNet18 adapted for grayscale multi-task classification."""

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.feature_extractor = models.resnet18(pretrained=False)
        self.feature_extractor.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        fc_in = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fc_in, 100)
        for i in range(n_tasks):
            setattr(self, f"task_{i}", nn.Linear(100, 10))

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.feature_extractor(x))
        outs = [getattr(self, f"task_{i}")(x) for i in range(self.n_tasks)]
        return torch.stack(outs, dim=1)  # (B, n_tasks, 10)
