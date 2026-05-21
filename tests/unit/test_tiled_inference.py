from wally_ai_api.utils.tiled_inference import (
    RawDetection,
    compute_tile_windows,
    nms_detections,
)


def test_compute_tile_windows_covers_image() -> None:
    windows = compute_tile_windows(512, 512, tile_size=256, overlap=64)
    assert len(windows) >= 4
    assert all(window.width > 0 and window.height > 0 for window in windows)


def test_nms_removes_overlapping_boxes() -> None:
    detections = [
        RawDetection(class_id=0, confidence=0.9, x1=10, y1=10, x2=50, y2=50),
        RawDetection(class_id=0, confidence=0.8, x1=12, y1=12, x2=48, y2=48),
        RawDetection(class_id=0, confidence=0.7, x1=200, y1=200, x2=240, y2=240),
    ]
    kept = nms_detections(detections, iou_threshold=0.45)
    assert len(kept) == 2
