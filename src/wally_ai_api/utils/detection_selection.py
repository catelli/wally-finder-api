from wally_ai_api.domain.entities.detection import Detection


def select_primary_detections(
    detections: list[Detection],
    *,
    max_count: int = 1,
    min_confidence: float = 0.88,
) -> list[Detection]:
    if not detections or max_count <= 0:
        return []

    ranked = sorted(detections, key=lambda item: item.confidence, reverse=True)
    selected: list[Detection] = []

    for candidate in ranked:
        if candidate.confidence < min_confidence:
            break
        selected.append(candidate)
        if len(selected) >= max_count:
            break

    return selected
