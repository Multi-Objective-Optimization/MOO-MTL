from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodOutput:
    weight_vector: Any
    loss_total: Any
    metadata: dict = field(default_factory=dict)


class MOOMethod(ABC):
    """Base class for all MOO methods in both synthetic and MTL domains.

    To add a new method:
      1. Subclass MOOMethod in the appropriate domain (synthetic/methods/ or mtl/methods/)
      2. Implement get_descent_direction() and step()
      3. Register in the domain's METHOD_REGISTRY
      4. Point to it via method.name in the config YAML
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def get_descent_direction(
        self,
        grads: Any,
        losses: Any,
        ref_vec: Any = None,
        pref_idx: int = None,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        """Compute task weight vector from gradients and losses.

        Returns:
            weight_vector: task weights (numpy array or torch.Tensor)
            metadata: dict of diagnostics (method-specific)
        """
        ...

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> MethodOutput:
        """Perform one optimization step. Returns MethodOutput."""
        ...
