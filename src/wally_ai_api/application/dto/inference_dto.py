from dataclasses import dataclass

from wally_ai_api.domain.entities.detection import Detection


@dataclass(frozen=True)
class InferenceResultDto:
    request_id: str
    detections: list[Detection]
    annotated_image: bytes
    content_type: str = "image/jpeg"
