from wally_ai_api.domain.entities.detection import BoundingBox, Detection  # noqa: F401
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
        min_confidence=0.72,
    )
    assert len(selected) == 1
    assert selected[0].confidence == 0.99


def test_returns_up_to_four_spatially_diverse_detections() -> None:
    base = _det(0.9)
    spread = [
        Detection(
            class_id=0,
            label="wally",
            confidence=0.99,
            bbox=BoundingBox(x1=0, y1=0, x2=256, y2=256),
        ),
        Detection(
            class_id=0,
            label="wally",
            confidence=0.95,
            bbox=BoundingBox(x1=512, y1=0, x2=768, y2=256),
        ),
        Detection(
            class_id=0,
            label="wally",
            confidence=0.91,
            bbox=BoundingBox(x1=0, y1=512, x2=256, y2=768),
        ),
        Detection(
            class_id=0,
            label="wally",
            confidence=0.9,
            bbox=BoundingBox(x1=512, y1=512, x2=768, y2=768),
        ),
        base,
    ]
    selected = select_primary_detections(spread, max_count=4, min_confidence=0.72)
    assert len(selected) == 4


def test_skips_overlapping_lower_confidence_candidate() -> None:
    selected = select_primary_detections(
        [
            Detection(
                class_id=0,
                label="wally",
                confidence=0.99,
                bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300),
            ),
            Detection(
                class_id=0,
                label="wally",
                confidence=0.95,
                bbox=BoundingBox(x1=120, y1=120, x2=280, y2=280),
            ),
        ],
        max_count=4,
        min_confidence=0.72,
        diversity_iou=0.25,
    )
    assert len(selected) == 1


def test_returns_empty_when_below_min_confidence() -> None:
    selected = select_primary_detections([_det(0.7)], min_confidence=0.72)
    assert selected == []
