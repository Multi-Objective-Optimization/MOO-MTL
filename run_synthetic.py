"""Synthetic MOO experiment runner.

Usage:
    python run_synthetic.py --config configs/synthetic/pmtl.yaml
    python run_synthetic.py --config configs/synthetic/cpmtl.yaml
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import yaml

from src.synthetic.methods import build_method
from src.synthetic.utils import circle_points
from src.synthetic.visualization import plot_pareto_front


PROBLEM_REGISTRY = {}


def _get_problem(name: str):
    if name == "concave":
        from src.synthetic.problems.concave import ConcaveProblem
        return ConcaveProblem()
    elif name == "zdt2":
        from src.synthetic.problems.zdt2 import Zdt2Variant
        return Zdt2Variant()
    else:
        raise ValueError(f"Unknown problem '{name}'. Available: concave | zdt2")


def run(cfg: dict) -> None:
    exp_cfg = cfg["experiment"]
    problem_cfg = cfg["problem"]
    method_cfg = cfg["method"]
    opt_cfg = cfg["optimization"]

    logging.basicConfig(
        level=getattr(logging, exp_cfg.get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    problem = _get_problem(problem_cfg["name"])
    method = build_method(method_cfg)
    n_pref = method_cfg["n_pref"]
    ref_vecs = circle_points([1], [n_pref])[0]
    n_dim = problem_cfg.get("n_dim", 20)
    step_size = opt_cfg.get("step_size", 1.0)
    n_iter = opt_cfg.get("n_iter", 100)
    init_fraction = method_cfg.get("init_fraction", 0.2)
    n_init = int(n_iter * init_fraction)

    pf = problem.pareto_front()
    solutions = []

    for i in range(n_pref):
        log.info(f"Preference vector {i + 1}/{n_pref}: {ref_vecs[i]}")
        x = np.random.uniform(-0.5, 0.5, n_dim)

        method_name = method_cfg["name"]

        if method_name == "ParetoMTL":
            for _ in range(n_init):
                x, f, _ = method.step(x, problem, ref_vecs, i, step_size, init_phase=True)
            for _ in range(n_iter - n_init):
                x, f, _ = method.step(x, problem, ref_vecs, i, step_size, init_phase=False)
        else:
            for _ in range(n_iter):
                x, f, _ = method.step(x, problem, ref_vecs, i, step_size)

        solutions.append(f)
        log.info(f"  f = {f}")

    solutions = np.array(solutions)
    log.info(f"Done. {n_pref} solutions found.")

    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if exp_cfg.get("save_plot", False):
        save_path = str(out_dir / "pareto_front.png")
        plot_pareto_front(pf, solutions, title=exp_cfg.get("name", ""), save_path=save_path)
        log.info(f"Plot saved to {save_path}")
    else:
        plot_pareto_front(pf, solutions, title=exp_cfg.get("name", ""))
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
