from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSchema(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBoxSchema


class AnnotatedImageSchema(BaseModel):
    content_type: str
    data_base64: str


class InferenceResponseSchema(BaseModel):
    request_id: str
    wally_found: bool
    detection_count: int
    detections: list[DetectionSchema]
    annotated_image: AnnotatedImageSchema


class InferenceImageResponseHeaders(BaseModel):
    request_id: str = Field(description="Request identifier")
    detection_count: str = Field(description="Number of detections")
