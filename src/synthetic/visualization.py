from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_pareto_front(
    pf: np.ndarray,
    solutions: np.ndarray,
    title: str = "",
    save_path: str = None,
) -> plt.Figure:
    """Plot ground truth Pareto front alongside achieved solutions.

    Args:
        pf: ground truth Pareto front, shape (n_points, 2)
        solutions: achieved objective values, shape (n_solutions, 2)
        title: plot title
        save_path: if given, save figure to this path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pf[:, 0], pf[:, 1], "k-.", label="Pareto Front", linewidth=1.5)
    ax.scatter(solutions[:, 0], solutions[:, 1], c="r", s=80, zorder=5, label="Solutions")
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
