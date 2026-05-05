from abc import ABC, abstractmethod
import logging


class SyntheticTrainer(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> None: ...
