from pathlib import Path

from ultralytics import YOLO

from wally_ai_api.domain.exceptions.model_errors import ModelNotLoadedError


class ModelLoader:
    def __init__(self, weights_path: Path) -> None:
        self._weights_path = weights_path
        self._model: YOLO | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> YOLO:
        if not self._weights_path.is_file():
            raise ModelNotLoadedError(f"Weights not found: {self._weights_path}")
        self._model = YOLO(str(self._weights_path))
        return self._model

    def get_model(self) -> YOLO:
        if self._model is None:
            raise ModelNotLoadedError()
        return self._model
