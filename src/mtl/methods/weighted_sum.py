# Weighted Sum baseline for MTL (Phase 1 of CPMTL pipeline).
# From: Efficient Continuous Pareto Exploration in Multi-Task Learning
#       Pingchuan Ma, Tao Du, Wojciech Matusik, ICML 2020

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored

from src.core.base_method import MOOMethod, MethodOutput
from src.mtl.metrics import evaluate_cpmtl


class WeightedSum(MOOMethod):
    """Fixed weighted sum scalarization. Uses preference vector directly as task weights."""

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, **kwargs):
        weight = ref_vec[pref_idx] if ref_vec is not None else torch.ones(2) / 2
        return weight, {}

    def step(self, network, optimizer, lr_scheduler, batch, pref_weight, closures):
        images, labels = batch
        device = next(network.parameters()).device
        images, labels = images.to(device), labels.to(device)

        logits = network(images)
        losses = [c(network, logits, labels) for c in closures]
        loss = sum(w * l for w, l in zip(pref_weight, losses))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        return MethodOutput(
            weight_vector=pref_weight,
            loss_total=loss.item(),
            metadata={"task_losses": [l.item() for l in losses]},
        )

    def train_epoch(self, network, trainloader, optimizer, lr_scheduler, pref_weight, closures):
        network.train(True)
        for batch in trainloader:
            self.step(network, optimizer, lr_scheduler, batch, pref_weight, closures)

    def evaluate(self, network, testloader, device, closures, header=""):
        losses, top1s = evaluate_cpmtl(network, testloader, device, closures)
        loss_msg = "[{}]".format("/".join([f"{l:.6f}" for l in losses]))
        top1_msg = "[{}]".format("/".join([f"{t * 100:.2f}%" for t in top1s]))
        msgs = [f"{header}:" if header else "", "loss", colored(loss_msg, "yellow"), "top@1", colored(top1_msg, "yellow")]
        print(" ".join(msgs))
        return losses, top1s
