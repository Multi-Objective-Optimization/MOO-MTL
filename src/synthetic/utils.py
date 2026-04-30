# Preference vector utilities for synthetic MOO experiments.

from itertools import product
import numpy as np


def circle_points(radii: list, counts: list) -> list:
    """Generate evenly distributed preference vectors on 2D quarter circle.

    Args:
        radii: list of circle radii (usually [1])
        counts: list of point counts per radius

    Returns:
        list of arrays, each shape (n, 2)
    """
    circles = []
    for r, n in zip(radii, counts):
        t = np.linspace(0, 0.5 * np.pi, n)
        circles.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    return circles


def evenly_dist_weights(num_weights: int, dim: int) -> list:
    """Generate evenly distributed weights on the (dim-1)-simplex interior.

    Excludes corners (0 or 1 values). Used in CPMTL / WeightedSum.

    Args:
        num_weights: number of points per dimension
        dim: number of tasks

    Returns:
        list of tuples, each summing to 1.0
    """
    return [
        ret for ret in product(np.linspace(0.0, 1.0, num_weights), repeat=dim)
        if round(sum(ret), 3) == 1.0 and all(r not in (0.0, 1.0) for r in ret)
    ]
