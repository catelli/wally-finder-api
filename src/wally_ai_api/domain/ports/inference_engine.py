from abc import ABC, abstractmethod

from wally_ai_api.domain.entities.detection import Detection
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.entities.model import ModelMetadata


class InferenceEnginePort(ABC):
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: ImageInput) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError
