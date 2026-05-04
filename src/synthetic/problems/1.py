# Concave synthetic MOO problem.
# From: Pareto Multi-Task Learning, Lin et al., NeurIPS 2019

import autograd.numpy as np
from autograd import grad


def _f1(x):
    n = len(x)
    return 1 - np.exp(-np.sum([(x[i] - 1.0 / np.sqrt(n)) ** 2 for i in range(n)]))


def _f2(x):
    n = len(x)
    return 1 - np.exp(-np.sum([(x[i] + 1.0 / np.sqrt(n)) ** 2 for i in range(n)]))


_f1_dx = grad(_f1)
_f2_dx = grad(_f2)


class ConcaveProblem():
    """Two-objective concave problem with analytic Pareto front."""

    def evaluate(self, x):
        return (
            np.stack([_f1(x), _f2(x)]),
            np.stack([_f1_dx(x), _f2_dx(x)]),
        )

    def pareto_front(self, n_points: int = 50):
        ps = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2), n_points)
        pf = []
        for x1 in ps:
            x = np.array([x1, x1])
            f, _ = self.evaluate(x)
            pf.append(f)
        return np.array(pf)
