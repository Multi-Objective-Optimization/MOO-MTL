import numpy as np
from pathlib import Path

from src.synthetic.problems import build_problem
from src.synthetic.methods import build_method
from src.synthetic.utils import circle_points
from src.synthetic.visualization import plot_pareto_front
from .base_trainer import SyntheticTrainer


class StandardSyntheticTrainer(SyntheticTrainer):
    """Generic trainer for all synthetic MOO methods.

    Handles init_phase automatically via init_fraction config key.
    Methods that do not use init_phase simply ignore the kwarg.
    """

    def run(self) -> None:
        cfg = self.cfg
        problem_cfg = cfg["problem"]
        method_cfg = cfg["method"]
        opt_cfg = cfg["optimization"]
        exp_cfg = cfg["experiment"]

        problem = build_problem(problem_cfg["name"])
        method = build_method(method_cfg)

        n_pref = method_cfg["n_pref"]
        ref_vecs = circle_points([1], [n_pref])[0]
        n_dim = problem_cfg.get("n_dim", 20)
        step_size = opt_cfg.get("step_size", 1.0)
        n_iter = opt_cfg.get("n_iter", 100)
        n_init = int(n_iter * method_cfg.get("init_fraction", 0.0))

        pf = problem.pareto_front()
        solutions = []

        for i in range(n_pref):
            self.log.info(f"Preference vector {i + 1}/{n_pref}: {ref_vecs[i]}")
            x = np.random.uniform(-0.5, 0.5, n_dim)

            for t in range(n_iter):
                x, f, _ = method.step(
                    x, problem, ref_vecs, i, step_size, init_phase=(t < n_init)
                )

            solutions.append(f)
            self.log.info(f"  f = {f}")

        solutions = np.array(solutions)
        self.log.info(f"Done. {n_pref} solutions found.")

        out_dir = Path(exp_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        if exp_cfg.get("save_plot", False):
            save_path = str(out_dir / "pareto_front.png")
            plot_pareto_front(pf, solutions, title=exp_cfg.get("name", ""), save_path=save_path)
            self.log.info(f"Plot saved to {save_path}")
        else:
            import matplotlib.pyplot as plt
            plot_pareto_front(pf, solutions, title=exp_cfg.get("name", ""))
            plt.show()
