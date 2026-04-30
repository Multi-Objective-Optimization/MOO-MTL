# CPMTL: Continuous Pareto Multi-Task Learning (Phase 2).
# From: Efficient Continuous Pareto Exploration in Multi-Task Learning
#       Pingchuan Ma, Tao Du, Wojciech Matusik, ICML 2020
#
# Requires a pre-trained checkpoint from WeightedSum (Phase 1).
# Uses KKT + Hessian-Vector Products to explore the Pareto front.

import numpy as np
import torch
from termcolor import colored

from src.core.base_method import MOOMethod, MethodOutput
from src.core.solvers.hvp_solver import VisionHVPSolver
from src.core.solvers.kkt_solver import MINRESKKTSolver, CGKKTSolver
from src.mtl.metrics import evaluate_cpmtl


class CPMTL(MOOMethod):
    """Continuous Pareto MTL using KKT-based exploration.

    This method takes a pre-trained (weighted sum) model and explores
    the Pareto front using second-order (HVP + KKT) updates.

    cfg keys used:
        kkt_solver (str): 'minres' | 'cg'
        n_steps (int): KKT steps per preference vector
        shift (float): MINRES shift parameter
        tol (float): solver tolerance
        damping (float): Hessian damping
        maxiter (int): max Krylov iterations
        stochastic (bool): stochastic HVP
        kkt_momentum (float): momentum for KKT
    """

    def get_descent_direction(self, grads, losses, ref_vec=None, pref_idx=None, **kwargs):
        raise NotImplementedError("CPMTL uses kkt_solver.backward() directly in step().")

    def step(self, network, optimizer, kkt_solver, beta):
        """One KKT-based Pareto exploration step.

        Args:
            network: PyTorch model
            optimizer: SGD/Adam optimizer
            kkt_solver: pre-built KKTSolver instance
            beta: preference direction tensor, shape (n_tasks,)
        """
        network.train(True)
        optimizer.zero_grad()
        kkt_solver.backward(beta, verbose=self.config.get("verbose", False))
        optimizer.step()
        return MethodOutput(weight_vector=beta, loss_total=None, metadata={})

    def build_kkt_solver(self, network, device, trainloader, closures):
        """Build HVP + KKT solver from config."""
        cfg = self.config
        hvp_solver = VisionHVPSolver(
            network, device, trainloader, closures,
            shared=cfg.get("shared", False),
        )
        hvp_solver.set_grad(batch=False)
        hvp_solver.set_hess(batch=True)

        solver_type = cfg.get("kkt_solver", "minres")
        common = dict(
            stochastic=cfg.get("stochastic", False),
            kkt_momentum=cfg.get("kkt_momentum", 0.0),
            create_graph=cfg.get("create_graph", False),
            grad_correction=cfg.get("grad_correction", False),
            tol=cfg.get("tol", 1e-5),
            damping=cfg.get("damping", 0.1),
            maxiter=cfg.get("maxiter", 50),
        )
        if solver_type == "minres":
            kkt_solver = MINRESKKTSolver(
                network, hvp_solver, device,
                shift=cfg.get("shift", 0.0),
                **common,
            )
        elif solver_type == "cg":
            kkt_solver = CGKKTSolver(
                network, hvp_solver, device,
                pd_strict=cfg.get("pd_strict", True),
                **{k: v for k, v in common.items() if k != "maxiter"},
                maxiter=cfg.get("maxiter", 5),
            )
        else:
            raise ValueError(f"Unknown kkt_solver '{solver_type}'. Options: minres | cg")

        return hvp_solver, kkt_solver

    def evaluate(self, network, testloader, device, closures, header=""):
        losses, top1s = evaluate_cpmtl(network, testloader, device, closures)
        loss_msg = "[{}]".format("/".join([f"{l:.6f}" for l in losses]))
        top1_msg = "[{}]".format("/".join([f"{t * 100:.2f}%" for t in top1s]))
        msgs = [f"{header}:" if header else "", "loss", colored(loss_msg, "yellow"), "top@1", colored(top1_msg, "yellow")]
        print(" ".join(msgs))
        return losses, top1s
