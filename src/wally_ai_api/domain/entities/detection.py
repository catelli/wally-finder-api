from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
