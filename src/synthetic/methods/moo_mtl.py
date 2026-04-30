# MOO-MTL for synthetic problems (NumPy/autograd).
# From: Multi-Task Learning as Multi-Objective Optimization
#       Ozan Sener, Vladlen Koltun, NeurIPS 2018

import numpy as np

from src.core.base_method import MOOMethod, MethodOutput
from src.core.solvers.min_norm_solver_numpy import MinNormSolver


class MOOMTLSynthetic(MOOMethod):
    """MOO-MTL for synthetic optimization (NumPy-based)."""

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, **kwargs):
        sol, _ = MinNormSolver.find_min_norm_element(grads)
        return sol, {}

    def step(self, x, problem, ref_vec=None, pref_idx=None, step_size=1.0):
        f, f_dx = problem.evaluate(x)
        weight, meta = self.get_descent_direction(f_dx, f)
        x = x - step_size * np.dot(weight.T, f_dx).flatten()
        return x, f, MethodOutput(weight_vector=weight, loss_total=None, metadata=meta)
