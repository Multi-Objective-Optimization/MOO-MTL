# ParetoMTL for multi-task learning (PyTorch).
# From: Pareto Multi-Task Learning, Lin et al., NeurIPS 2019

import numpy as np
import torch
from torch.autograd import Variable

from core.base_method import MOOMethod, MethodOutput
from core.solvers.min_norm_solver_torch import MinNormSolver


class ParetoMTL(MOOMethod):
    """ParetoMTL for neural network MTL (PyTorch-based)."""

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, init_phase=False, **kwargs):
        if init_phase:
            flag, weight = _get_d_paretomtl_init(grads, losses, ref_vec, pref_idx)
            return weight, {"flag": flag}
        weight = _get_d_paretomtl(grads, losses, ref_vec, pref_idx)
        return weight, {}

    def step(self, model, optimizer, batch, ref_vec, pref_idx, n_tasks, init_phase=False):
        X, ts = batch
        device = next(model.parameters()).device
        X, ts = X.to(device), ts.to(device)

        grads = {}
        losses_vec = []
        for i in range(n_tasks):
            optimizer.zero_grad()
            task_loss = model(X, ts)
            losses_vec.append(task_loss[i].data)
            task_loss[i].backward()
            grads[i] = [
                Variable(p.grad.data.clone().flatten(), requires_grad=False)
                for p in model.parameters() if p.grad is not None
            ]

        grads_tensor = torch.stack([torch.cat(grads[i]) for i in range(n_tasks)])
        losses_tensor = torch.stack(losses_vec)

        weight, meta = self.get_descent_direction(
            grads_tensor, losses_tensor, ref_vec, pref_idx, init_phase=init_phase
        )

        if meta.get("flag", False):
            return MethodOutput(weight_vector=weight, loss_total=None, metadata=meta)

        if not init_phase:
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight))
            weight = weight * normalize_coeff

        optimizer.zero_grad()
        task_loss = model(X, ts)
        loss_total = sum(weight[i] * task_loss[i] for i in range(n_tasks))
        loss_total.backward()
        optimizer.step()

        return MethodOutput(weight_vector=weight, loss_total=loss_total.item(), metadata=meta)


def _get_d_paretomtl_init(grads, value, weights, i):
    flag = False
    nobj = value.shape

    current_weight = weights[i]
    w = weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).to(grads.device).float()
    else:
        vec = torch.matmul(w[idx], grads)
        sol, _ = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

    weight0 = torch.sum(torch.stack([sol[j] * w[idx][j, 0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 = torch.sum(torch.stack([sol[j] * w[idx][j, 1] for j in torch.arange(0, torch.sum(idx))]))
    return flag, torch.stack([weight0, weight1])


def _get_d_paretomtl(grads, value, weights, i):
    current_weight = weights[i]
    w = weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    if torch.sum(idx) <= 0:
        sol, _ = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).to(grads.device).float()

    vec = torch.cat((grads, torch.matmul(w[idx], grads)))
    sol, _ = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

    n_active = torch.sum(idx)
    weight0 = sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2, 0] for j in torch.arange(2, 2 + n_active)]))
    weight1 = sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2, 1] for j in torch.arange(2, 2 + n_active)]))
    return torch.stack([weight0, weight1])
