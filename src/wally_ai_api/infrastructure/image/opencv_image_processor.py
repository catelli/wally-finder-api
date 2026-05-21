from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from wally_ai_api.core.security import ALLOWED_EXTENSIONS
from wally_ai_api.domain.entities.detection import Detection
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.exceptions.image_errors import ImageValidationError
from wally_ai_api.domain.ports.image_processor import ImageProcessorPort


class OpenCvImageProcessor(ImageProcessorPort):
    def validate(
        self,
        image: ImageInput,
        max_bytes: int,
        allowed_types: list[str],
    ) -> None:
        if not image.content:
            raise ImageValidationError("Empty image payload", code="empty_image")
        if len(image.content) > max_bytes:
            raise ImageValidationError(
                f"Image exceeds max size of {max_bytes} bytes",
                code="image_too_large",
            )
        if image.content_type not in allowed_types:
            raise ImageValidationError(
                f"Unsupported content type: {image.content_type}",
                code="unsupported_media_type",
            )

        extension = PathLikeExtension.from_filename(image.filename)
        if extension and extension not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file extension: {extension}",
                code="unsupported_extension",
            )

        try:
            pil_image = Image.open(io.BytesIO(image.content))
            pil_image.verify()
        except Exception as exc:
            raise ImageValidationError("Invalid or corrupted image file") from exc

    def draw_detections(self, image: ImageInput, detections: list[Detection]) -> bytes:
        array = np.frombuffer(image.content, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ImageValidationError("Unable to decode image for annotation")

        for detection in detections:
            box = detection.bbox
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not success:
            raise ImageValidationError("Failed to encode annotated image")
        return encoded.tobytes()


class PathLikeExtension:
    @staticmethod
    def from_filename(filename: str) -> str | None:
        if "." not in filename:
            return None
        return "." + filename.rsplit(".", 1)[-1].lower()
