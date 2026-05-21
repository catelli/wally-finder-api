from wally_ai_api.core.config import get_model_settings
from wally_ai_api.core.constants import DEFAULT_CLASS_LABEL
from wally_ai_api.domain.entities.detection import BoundingBox, Detection
from wally_ai_api.utils.tiled_inference import RawDetection


def map_raw_detections(raw: list[RawDetection]) -> list[Detection]:
    class_names = get_model_settings().class_names
    mapped: list[Detection] = []
    for item in raw:
        label = (
            class_names[item.class_id]
            if 0 <= item.class_id < len(class_names)
            else DEFAULT_CLASS_LABEL
        )
        mapped.append(
            Detection(
                class_id=item.class_id,
                label=label,
                confidence=item.confidence,
                bbox=BoundingBox(
                    x1=item.x1,
                    y1=item.y1,
                    x2=item.x2,
                    y2=item.y2,
                ),
            )
        )
    return mapped
