from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wally_ai_api.core.config import get_model_settings
from wally_ai_api.domain.entities.detection import Detection
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.entities.model import ModelMetadata
from wally_ai_api.domain.exceptions.inference_errors import InferenceError
from wally_ai_api.domain.exceptions.model_errors import ModelNotLoadedError
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort
from wally_ai_api.domain.ports.model_repository import ModelRepositoryPort
from wally_ai_api.infrastructure.model.model_loader import ModelLoader
from wally_ai_api.infrastructure.model.prediction_mapper import map_raw_detections
from wally_ai_api.utils.tiled_inference import (
    RawDetection,
    compute_tile_windows,
    extract_tile,
    nms_detections,
    shift_boxes_to_global,
)


class YoloTiledInferenceEngine(InferenceEnginePort):
    def __init__(self, repository: ModelRepositoryPort) -> None:
        self._repository = repository
        self._loader = ModelLoader(repository.get_weights_path())
        self._settings = get_model_settings()

    def is_loaded(self) -> bool:
        return self._loader.is_loaded

    def load(self) -> None:
        if not self._repository.weights_exist():
            raise ModelNotLoadedError(
                f"Weights not found at {self._repository.get_weights_path()}"
            )
        self._loader.load()

    def warmup(self) -> None:
        if not self.is_loaded():
            self.load()
        blank = Image.new("RGB", (self._settings.imgsz, self._settings.imgsz), color=(0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="JPEG")
        sample = ImageInput(
            content=buffer.getvalue(),
            content_type="image/jpeg",
            filename="warmup.jpg",
        )
        self.predict(sample)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="wally_tiles",
            version="1.0.0",
            classes=list(self._settings.class_names),
            input_size=self._settings.imgsz,
            tiled_inference=True,
        )

    def predict(self, image: ImageInput) -> list[Detection]:
        if not self.is_loaded():
            raise ModelNotLoadedError()

        try:
            pil_image = Image.open(io.BytesIO(image.content)).convert("RGB")
        except Exception as exc:
            raise InferenceError("Unable to decode image for inference") from exc

        windows = compute_tile_windows(
            pil_image.width,
            pil_image.height,
            self._settings.tile_size,
            self._settings.tile_overlap,
        )
        model = self._loader.get_model()
        merged: list[RawDetection] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for index, window in enumerate(windows):
                tile = extract_tile(pil_image, window, self._settings.tile_size)
                tile_path = temp_path / f"tile_{index}.jpg"
                tile.save(tile_path, format="JPEG")

                results = model.predict(
                    source=str(tile_path),
                    imgsz=self._settings.imgsz,
                    conf=self._settings.confidence_threshold,
                    iou=self._settings.iou_threshold,
                    device=self._settings.device,
                    save=False,
                    verbose=False,
                )

                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                global_boxes = shift_boxes_to_global(boxes, window)

                for class_id, score, box in zip(classes, confidences, global_boxes):
                    merged.append(
                        RawDetection(
                            class_id=int(class_id),
                            confidence=float(score),
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                        )
                    )

        filtered = nms_detections(merged, iou_threshold=self._settings.merge_iou)
        return map_raw_detections(filtered)
