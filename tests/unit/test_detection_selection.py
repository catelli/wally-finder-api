from wally_ai_api.domain.entities.detection import BoundingBox, Detection
from wally_ai_api.utils.detection_selection import select_primary_detections


def _det(confidence: float) -> Detection:
    return Detection(
        class_id=0,
        label="wally",
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
    )


def test_returns_only_highest_confidence_detection() -> None:
    selected = select_primary_detections(
        [_det(0.5), _det(0.99), _det(0.9)],
        max_count=1,
        min_confidence=0.88,
    )
    assert len(selected) == 1
    assert selected[0].confidence == 0.99


def test_returns_empty_when_below_min_confidence() -> None:
    selected = select_primary_detections([_det(0.7)], min_confidence=0.88)
    assert selected == []
