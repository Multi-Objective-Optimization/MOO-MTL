# Linear Scalarization baseline for synthetic problems (NumPy/autograd).
# From: Pareto Multi-Task Learning, Lin et al., NeurIPS 2019

import numpy as np

from src.core.base_method import MOOMethod, MethodOutput


class LinearScalarizationSynthetic(MOOMethod):
    """Linear scalarization with randomly sampled preference weight."""

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, **kwargs):
        r = np.random.rand(1)
        weight = np.stack([r, 1 - r])
        return weight, {}

    def step(self, x, problem, ref_vec=None, pref_idx=None, step_size=1.0):
        f, f_dx = problem.evaluate(x)
        weight, meta = self.get_descent_direction(f_dx, f)
        x = x - step_size * np.dot(weight.T, f_dx).flatten()
        return x, f, MethodOutput(weight_vector=weight, loss_total=None, metadata=meta)
