# Min-norm element in convex hull of gradient vectors (PyTorch version).
# From: Multi-Task Learning as Multi-Objective Optimization
#       Ozan Sener, Vladlen Koltun, NeurIPS 2018
#       https://github.com/intel-isl/MultiObjectiveOptimization

import numpy as np
import torch


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k], vecs[j][k]).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k], vecs[i][k]).item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """Find minimum norm element in convex hull of vecs.

        vecs: list of lists of tensors (one list per task, one tensor per layer)
        Returns: (sol_vec, min_norm_squared)
        """
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        for _ in range(MinNormSolver.MAX_ITER):
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            v1v1 = v1v2 = v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            if np.sum(np.abs(new_sol_vec - sol_vec)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """Frank-Wolfe variant of find_min_norm_element."""
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        for _ in range(MinNormSolver.MAX_ITER):
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))
            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            if np.sum(np.abs(new_sol_vec - sol_vec)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        raise ValueError(f"Invalid normalization type: {normalization_type!r}")
    return gn


def find_min_norm_element(vecs):
    """Convenience wrapper that accepts a 2D tensor (n_tasks x n_params).

    Used by kkt_solver.py (CPMTL).
    """
    alphas, nd = MinNormSolver.find_min_norm_element([[v] for v in vecs])
    return alphas, nd
