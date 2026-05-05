from .base_trainer import MTLTrainer
from .pmtl_trainer import ParetoMTLTrainer
from .cpmtl_trainer import WeightedSumTrainer, CPMTLTrainer

MTL_TRAINER_REGISTRY = {
    "ParetoMTL": ParetoMTLTrainer,
    "WeightedSum": WeightedSumTrainer,
    "CPMTL": CPMTLTrainer,
}


def build_trainer(method_name: str, cfg: dict) -> MTLTrainer:
    if method_name not in MTL_TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown MTL method '{method_name}'. "
            f"Available: {list(MTL_TRAINER_REGISTRY)}"
        )
    return MTL_TRAINER_REGISTRY[method_name](cfg)
