from .base_trainer import SyntheticTrainer
from .standard_trainer import StandardSyntheticTrainer

SYNTHETIC_TRAINER_REGISTRY = {
    "ParetoMTL": StandardSyntheticTrainer,
    "MOOMTL": StandardSyntheticTrainer,
    "LinearScalarization": StandardSyntheticTrainer,
}


def build_trainer(method_name: str, cfg: dict) -> SyntheticTrainer:
    if method_name not in SYNTHETIC_TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown synthetic method '{method_name}'. "
            f"Available: {list(SYNTHETIC_TRAINER_REGISTRY)}"
        )
    return SYNTHETIC_TRAINER_REGISTRY[method_name](cfg)
