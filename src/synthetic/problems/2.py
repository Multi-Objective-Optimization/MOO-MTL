# ZDT2 variant synthetic MOO problem.
# From: Efficient Continuous Pareto Exploration in Multi-Task Learning
#       Pingchuan Ma, Tao Du, Wojciech Matusik, ICML 2020

import numpy as np


class Zdt2Variant():
    """ZDT2 variant with 3D decision space and 2 objectives.

    The problem is defined via a differentiable reparametrization that
    makes the Pareto set a 2D manifold in decision space.
    """

    def __init__(self) -> None:
        self.n = 3
        self.m = 2

    def _remap(self, x):
        x = np.asarray(x).ravel()
        x2 = np.zeros(self.n)
        x2[0] = np.sin(x[0] + x[1] ** 2 + x[2] ** 2) * 0.5 + 0.5
        s = np.sum(x[1:] ** 2)
        x2[1:] = 0.5 * np.cos(s) + 0.5
        return x2

    def _remap_grad(self, x):
        x = np.asarray(x).ravel()
        jac = np.zeros((self.n, self.n))
        jac[0] = 0.5 * np.cos(x[0] + x[1] ** 2 + x[2] ** 2) * np.array([1, 2 * x[1], 2 * x[2]])
        s = np.sum(x[1:] ** 2)
        g_s = np.zeros(self.n)
        g_s[1:] = 2 * x[1:]
        jac[1:] = -0.5 * np.sin(s) * g_s
        return jac

    def _f_inner(self, x):
        f1 = x[0]
        g = 1 + 9 / (self.n - 1) * np.sum(x[1:])
        f2 = g * (1 - (x[0] / g) ** 2)
        return np.array([f1, f2])

    def _grad_inner(self, x):
        g1 = np.zeros(self.n)
        g1[0] = 1
        grad_g = np.zeros(self.n)
        grad_g[1:] = 9 / (self.n - 1)
        g = 1 + 9 / (self.n - 1) * np.sum(x[1:])
        g2 = grad_g * (1 - (x[0] / g) ** 2)
        g2[0] += -2 * x[0] / g
        g2[1:] += 2 * (x[0] / g) ** 2 * grad_g[1:]
        return np.array([g1, g2])

    def objectives(self, x):
        return self._f_inner(self._remap(np.asarray(x).ravel()))

    def gradients(self, x):
        x = np.asarray(x).ravel()
        x_new = self._remap(x)
        grad_x_new = self._remap_grad(x)
        g1, g2 = self._grad_inner(x_new)
        return np.array([g1.T @ grad_x_new, g2.T @ grad_x_new])

    def evaluate(self, x):
        return self.objectives(x), self.gradients(x)

    def pareto_front(self, n_points: int = 101):
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1 - f1 ** 2
        return np.column_stack([f1, f2])

    def sample_pareto_set(self):
        """Sample a random point from the Pareto set."""
        x = np.zeros(self.n)
        x[0] = np.random.uniform(-np.pi / 2, np.pi / 2) - np.pi
        theta = np.random.uniform(-np.pi, np.pi)
        x[1] = np.sqrt(np.pi) * np.cos(theta)
        x[2] = np.sqrt(np.pi) * np.sin(theta)
        return x
