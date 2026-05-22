from wally_ai_api.domain.entities.detection import BoundingBox, Detection
from wally_ai_api.utils.grid_refinement import (
    dedupe_grid_cells,
    refine_wally_detections,
    snap_bbox_to_grid,
)


def _det(x1: float, y1: float, x2: float, y2: float, confidence: float) -> Detection:
    return Detection(
        class_id=0,
        label="wally",
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_snap_bbox_aligns_to_256_grid() -> None:
    bbox = snap_bbox_to_grid(2524.0, 431.0, 2746.0, 617.0, 256, 2800, 1760)
    assert bbox.x1 == 256 * 10
    assert bbox.y1 == 256 * 2
    assert bbox.x2 == 2800
    assert bbox.y2 == 256 * 3


def test_dedupe_keeps_highest_confidence_per_cell() -> None:
    detections = [
        _det(256, 256, 512, 512, 0.7),
        _det(300, 280, 480, 500, 0.95),
    ]
    deduped = dedupe_grid_cells(detections, 256)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.95


def test_refine_snaps_and_filters_oversized_boxes() -> None:
    refined = refine_wally_detections(
        [_det(0, 0, 2000, 1500, 0.9)],
        image_width=2800,
        image_height=1760,
        grid_size=256,
        snap_to_grid=True,
        cluster_iou=0.4,
        min_area_ratio=0.15,
        max_area_ratio=1.75,
    )
    assert len(refined) == 0
