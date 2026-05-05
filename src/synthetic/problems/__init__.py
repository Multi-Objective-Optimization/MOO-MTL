from .lin2019_ex1 import Lin2019Ex1
from .ma2020_zdt2 import Ma2020ZDT2

PROBLEM_REGISTRY = {
    "lin2019_ex1": Lin2019Ex1,
    "ma2020_zdt2": Ma2020ZDT2,
}


def build_problem(name: str):
    if name not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem '{name}'. Available: {list(PROBLEM_REGISTRY)}"
        )
    return PROBLEM_REGISTRY[name]()
