import pytest

from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.exceptions.image_errors import ImageValidationError
from wally_ai_api.infrastructure.image.opencv_image_processor import OpenCvImageProcessor


def test_rejects_empty_payload() -> None:
    processor = OpenCvImageProcessor()
    image = ImageInput(content=b"", content_type="image/jpeg", filename="empty.jpg")
    with pytest.raises(ImageValidationError, match="Empty"):
        processor.validate(image, max_bytes=1_000_000, allowed_types=["image/jpeg"])


def test_rejects_unsupported_content_type(valid_image_bytes: bytes) -> None:
    processor = OpenCvImageProcessor()
    image = ImageInput(
        content=valid_image_bytes,
        content_type="text/plain",
        filename="scene.jpg",
    )
    with pytest.raises(ImageValidationError, match="Unsupported content type"):
        processor.validate(image, max_bytes=1_000_000, allowed_types=["image/jpeg"])
