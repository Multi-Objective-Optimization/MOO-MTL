from .lenet import LeNetPMTL, LeNetCPMTL
from .resnet import ResNet18MTL
from .task_wrapper import MultiTaskWrapper

BACKBONE_REGISTRY = {
    "lenet": LeNetPMTL,
    "lenet_cpmtl": LeNetCPMTL,
    "resnet18": ResNet18MTL,
}


def build_model(model_cfg: dict) -> MultiTaskWrapper:
    """Build a MultiTaskWrapper from config.

    model_cfg keys:
        backbone (str): 'lenet' | 'lenet_cpmtl' | 'resnet18'
        n_tasks (int): number of tasks
        init_weight (list, optional): initial task weights
    """
    name = model_cfg["backbone"]
    n_tasks = model_cfg["n_tasks"]
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY)}")
    backbone_cls = BACKBONE_REGISTRY[name]
    backbone = backbone_cls() if name == "lenet_cpmtl" else backbone_cls(n_tasks)
    return MultiTaskWrapper(backbone, n_tasks, model_cfg.get("init_weight"))
