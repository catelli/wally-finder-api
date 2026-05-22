from wally_ai_api.domain.entities.detection import Detection
from wally_ai_api.utils.tiled_inference import iou


def select_primary_detections(
    detections: list[Detection],
    *,
    max_count: int = 4,
    min_confidence: float = 0.72,
    diversity_iou: float = 0.25,
) -> list[Detection]:
    if not detections or max_count <= 0:
        return []

    ranked = sorted(detections, key=lambda item: item.confidence, reverse=True)
    selected: list[Detection] = []

    for candidate in ranked:
        if candidate.confidence < min_confidence:
            break

        candidate_box = [
            candidate.bbox.x1,
            candidate.bbox.y1,
            candidate.bbox.x2,
            candidate.bbox.y2,
        ]
        overlaps_selected = any(
            iou(
                candidate_box,
                [
                    kept.bbox.x1,
                    kept.bbox.y1,
                    kept.bbox.x2,
                    kept.bbox.y2,
                ],
            )
            >= diversity_iou
            for kept in selected
        )
        if overlaps_selected:
            continue

        selected.append(candidate)
        if len(selected) >= max_count:
            break

    return selected
