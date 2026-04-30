from core.base_method import MOOMethod
from .pareto_mtl import ParetoMTLSynthetic
from .moo_mtl import MOOMTLSynthetic
from .linear_scalarization import LinearScalarizationSynthetic

SYNTHETIC_METHOD_REGISTRY = {
    "ParetoMTL": ParetoMTLSynthetic,
    "MOOMTL": MOOMTLSynthetic,
    "LinearScalarization": LinearScalarizationSynthetic,
}


def build_method(method_cfg: dict) -> MOOMethod:
    name = method_cfg["name"]
    if name not in SYNTHETIC_METHOD_REGISTRY:
        raise ValueError(
            f"Unknown synthetic method '{name}'. "
            f"Available: {list(SYNTHETIC_METHOD_REGISTRY)}"
        )
    return SYNTHETIC_METHOD_REGISTRY[name](method_cfg)
