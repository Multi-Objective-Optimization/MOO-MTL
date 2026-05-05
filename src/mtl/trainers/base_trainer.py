from abc import ABC, abstractmethod
import logging


class MTLTrainer(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def train(self, pref_idx: int = None) -> None: ...
