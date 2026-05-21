from wally_ai_api.application.dto.inference_dto import InferenceResultDto
from wally_ai_api.application.use_cases.validate_image import ValidateImageUseCase
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.ports.image_processor import ImageProcessorPort
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort
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
        detections = self._inference_engine.predict(image)
        annotated = self._image_processor.draw_detections(image, detections)
        return InferenceResultDto(
            request_id=new_request_id(),
            detections=detections,
            annotated_image=annotated,
            content_type="image/jpeg",
        )
