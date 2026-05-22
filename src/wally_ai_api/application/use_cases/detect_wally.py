import io

from PIL import Image

from wally_ai_api.application.dto.inference_dto import InferenceResultDto
from wally_ai_api.application.use_cases.validate_image import ValidateImageUseCase
from wally_ai_api.core.config import get_model_settings
from wally_ai_api.domain.entities.image import ImageInput
from wally_ai_api.domain.ports.image_processor import ImageProcessorPort
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort
from wally_ai_api.utils.detection_selection import select_primary_detections
from wally_ai_api.utils.grid_refinement import refine_wally_detections
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
        pil_image = Image.open(io.BytesIO(image.content))
        refined = refine_wally_detections(
            raw_detections,
            pil_image.width,
            pil_image.height,
            grid_size=settings.grid_cell_size,
            snap_to_grid=settings.snap_to_grid,
            cluster_iou=settings.cluster_merge_iou,
            min_area_ratio=settings.min_box_area_ratio,
            max_area_ratio=settings.max_box_area_ratio,
        )
        detections = select_primary_detections(
            refined,
            max_count=settings.max_output_detections,
            min_confidence=settings.min_output_confidence,
            diversity_iou=settings.selection_diversity_iou,
        )
        annotated = self._image_processor.draw_detections(image, detections)
        return InferenceResultDto(
            request_id=new_request_id(),
            detections=detections,
            annotated_image=annotated,
            content_type="image/jpeg",
        )
