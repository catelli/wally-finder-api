from wally_ai_api.infrastructure.model.prediction_mapper import map_raw_detections
from wally_ai_api.utils.tiled_inference import RawDetection


def test_maps_class_label_from_settings() -> None:
    mapped = map_raw_detections(
        [RawDetection(class_id=0, confidence=0.88, x1=1, y1=2, x2=3, y2=4)]
    )
    assert len(mapped) == 1
    assert mapped[0].label == "wally"
    assert mapped[0].confidence == 0.88
    assert mapped[0].bbox.x2 == 3
