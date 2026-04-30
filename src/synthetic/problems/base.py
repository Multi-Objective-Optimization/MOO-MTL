from abc import ABC, abstractmethod
import numpy as np


class SyntheticProblem(ABC):
    """Abstract base class for synthetic MOO problems."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (objectives, gradients), each shape (n_tasks,) or (n_tasks, n_dim)."""
        ...

    @abstractmethod
    def pareto_front(self, n_points: int = 50) -> np.ndarray:
        """Return ground truth Pareto front, shape (n_points, n_tasks)."""
        ...
