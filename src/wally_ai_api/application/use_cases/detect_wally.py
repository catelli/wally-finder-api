from wally_ai_api.application.dto.inference_dto import InferenceResultDto
from wally_ai_api.application.use_cases.validate_image import ValidateImageUseCase
from wally_ai_api.core.config import get_model_settings
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.ports.image_processor import ImageProcessorPort
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort
from wally_ai_api.utils.detection_selection import select_primary_detections
from wally_ai_api.utils.id_utils import new_request_id


class DetectWallyUseCase:
    def __init__(
        self,
        inference_engine: InferenceEnginePort,
        image_processor: ImageProcessorPort,
    ) -> None:
        self._inference_engine = inference_engine
        self._image_processor = image_processor
        self._validate_image = ValidateImageUseCase(image_processor)

    def execute(self, image: ImageInput) -> InferenceResultDto:
        self._validate_image.execute(image)
        settings = get_model_settings()
        raw_detections = self._inference_engine.predict(image)
        detections = select_primary_detections(
            raw_detections,
            max_count=settings.max_output_detections,
            min_confidence=settings.min_output_confidence,
        )
        annotated = self._image_processor.draw_detections(image, detections)
        return InferenceResultDto(
            request_id=new_request_id(),
            detections=detections,
            annotated_image=annotated,
            content_type="image/jpeg",
        )
