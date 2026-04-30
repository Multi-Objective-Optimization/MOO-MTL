from src.core.base_method import MOOMethod
from .pareto_mtl import ParetoMTL
from .weighted_sum import WeightedSum
from .cpmtl import CPMTL

MTL_METHOD_REGISTRY = {
    "ParetoMTL": ParetoMTL,
    "WeightedSum": WeightedSum,
    "CPMTL": CPMTL,
}


def build_method(method_cfg: dict) -> MOOMethod:
    name = method_cfg["name"]
    if name not in MTL_METHOD_REGISTRY:
        raise ValueError(
            f"Unknown MTL method '{name}'. "
            f"Available: {list(MTL_METHOD_REGISTRY)}"
        )
    return MTL_METHOD_REGISTRY[name](method_cfg)
