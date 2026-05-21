from abc import ABC, abstractmethod

from wally_ai_api.domain.entities.detection import Detection
from wally_ai_api.domain.entities.image import ImageInput


class ImageProcessorPort(ABC):
    @abstractmethod
    def validate(self, image: ImageInput, max_bytes: int, allowed_types: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_detections(self, image: ImageInput, detections: list[Detection]) -> bytes:
        raise NotImplementedError
