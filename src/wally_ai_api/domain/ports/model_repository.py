from abc import ABC, abstractmethod
from pathlib import Path


class ModelRepositoryPort(ABC):
    @abstractmethod
    def get_weights_path(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def weights_exist(self) -> bool:
        raise NotImplementedError
