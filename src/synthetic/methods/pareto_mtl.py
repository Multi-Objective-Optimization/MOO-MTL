# ParetoMTL for synthetic problems (NumPy/autograd).
# From: Pareto Multi-Task Learning, Lin et al., NeurIPS 2019

import numpy as np

from src.core.base_method import MOOMethod, MethodOutput
from src.core.solvers.min_norm_solver_numpy import MinNormSolver


class ParetoMTLSynthetic(MOOMethod):
    """ParetoMTL for synthetic optimization (NumPy-based)."""

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, init_phase=False, **kwargs):
        if init_phase:
            weight = _get_d_paretomtl_init(grads, losses, ref_vec, pref_idx)
            return weight, {}
        weight = _get_d_paretomtl(grads, losses, ref_vec, pref_idx)
        return weight, {}

    def step(self, x, problem, ref_vec, pref_idx, step_size, init_phase=False):
        f, f_dx = problem.evaluate(x)
        weight, meta = self.get_descent_direction(f_dx, f, ref_vec, pref_idx, init_phase=init_phase)
        x = x - step_size * np.dot(weight.T, f_dx).flatten()
        return x, f, MethodOutput(weight_vector=weight, loss_total=None, metadata=meta)


def _get_d_paretomtl_init(grads, value, weights, i):
    nobj, dim = grads.shape

    # check active constraints# check active constraints
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = (
        np.delete(weights, i, axis=0)
        / np.linalg.norm(np.delete(weights, i, axis=0), axis=1, keepdims=True)
    )
    w = normalized_rest_weights - normalized_current_weight

    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec = np.dot(w[idx], grads)
        sol, _ = MinNormSolver.find_min_norm_element(vec)

    weight0 = np.sum([sol[j] * w[idx][j, 0] for j in range(np.sum(idx))])
    weight1 = np.sum([sol[j] * w[idx][j, 1] for j in range(np.sum(idx))])
    return np.stack([weight0, weight1])


def _get_d_paretomtl(grads, value, weights, i):
    nobj, dim = grads.shape
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = (
        np.delete(weights, i, axis=0)
        / np.linalg.norm(np.delete(weights, i, axis=0), axis=1, keepdims=True)
    )
    w = normalized_rest_weights - normalized_current_weight

    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    vec = np.concatenate((grads, np.dot(w[idx], grads)), axis=0)
    sol, _ = MinNormSolver.find_min_norm_element(vec)

    n_active = np.sum(idx)
    weight0 = sol[0] + np.sum([sol[j] * w[idx][j - 2, 0] for j in range(2, 2 + n_active)])
    weight1 = sol[1] + np.sum([sol[j] * w[idx][j - 2, 1] for j in range(2, 2 + n_active)])
    return np.stack([weight0, weight1])
