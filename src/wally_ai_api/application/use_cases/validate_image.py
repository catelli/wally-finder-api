from wally_ai_api.core.config import get_app_settings
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.ports.image_processor import ImageProcessorPort


class ValidateImageUseCase:
    def __init__(self, image_processor: ImageProcessorPort) -> None:
        self._image_processor = image_processor
        self._settings = get_app_settings()

    def execute(self, image: ImageInput) -> None:
        self._image_processor.validate(
            image,
            max_bytes=self._settings.max_upload_bytes,
            allowed_types=self._settings.allowed_content_types,
        )
