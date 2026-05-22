from __future__ import annotations

from wally_ai_api.domain.entities.detection import BoundingBox, Detection
from wally_ai_api.utils.tiled_inference import RawDetection, iou, nms_detections


def snap_bbox_to_grid(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    grid_size: int,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    col = max(0, int(center_x // grid_size))
    row = max(0, int(center_y // grid_size))
    snapped_x1 = col * grid_size
    snapped_y1 = row * grid_size
    snapped_x2 = min(snapped_x1 + grid_size, image_width)
    snapped_y2 = min(snapped_y1 + grid_size, image_height)
    return BoundingBox(
        x1=float(snapped_x1),
        y1=float(snapped_y1),
        x2=float(snapped_x2),
        y2=float(snapped_y2),
    )


def box_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def grid_cell_key(bbox: BoundingBox, grid_size: int) -> tuple[int, int]:
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    return (int(center_y // grid_size), int(center_x // grid_size))


def merge_weighted_cluster(detections: list[Detection]) -> Detection:
    total_weight = sum(item.confidence for item in detections)
    if total_weight <= 0:
        best = max(detections, key=lambda item: item.confidence)
        return best

    x1 = sum(item.bbox.x1 * item.confidence for item in detections) / total_weight
    y1 = sum(item.bbox.y1 * item.confidence for item in detections) / total_weight
    x2 = sum(item.bbox.x2 * item.confidence for item in detections) / total_weight
    y2 = sum(item.bbox.y2 * item.confidence for item in detections) / total_weight
    confidence = max(item.confidence for item in detections)
    anchor = max(detections, key=lambda item: item.confidence)
    return Detection(
        class_id=anchor.class_id,
        label=anchor.label,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def cluster_merge_detections(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    clusters: list[list[Detection]] = []

    for candidate in ordered:
        candidate_box = [
            candidate.bbox.x1,
            candidate.bbox.y1,
            candidate.bbox.x2,
            candidate.bbox.y2,
        ]
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            anchor_box = [
                anchor.bbox.x1,
                anchor.bbox.y1,
                anchor.bbox.x2,
                anchor.bbox.y2,
            ]
            if iou(candidate_box, anchor_box) >= iou_threshold:
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])

    return [merge_weighted_cluster(cluster) for cluster in clusters]


def filter_by_box_area(
    detections: list[Detection],
    grid_size: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[Detection]:
    grid_area = float(grid_size * grid_size)
    filtered: list[Detection] = []
    for detection in detections:
        ratio = box_area(detection.bbox) / grid_area
        if min_area_ratio <= ratio <= max_area_ratio:
            filtered.append(detection)
    return filtered


def snap_detections_to_grid(
    detections: list[Detection],
    grid_size: int,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    snapped: list[Detection] = []
    for detection in detections:
        bbox = snap_bbox_to_grid(
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
            grid_size,
            image_width,
            image_height,
        )
        snapped.append(
            Detection(
                class_id=detection.class_id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=bbox,
            )
        )
    return snapped


def dedupe_grid_cells(detections: list[Detection], grid_size: int) -> list[Detection]:
    best_by_cell: dict[tuple[int, int], Detection] = {}
    for detection in detections:
        cell = grid_cell_key(detection.bbox, grid_size)
        existing = best_by_cell.get(cell)
        if existing is None or detection.confidence > existing.confidence:
            best_by_cell[cell] = detection
    return sorted(best_by_cell.values(), key=lambda item: item.confidence, reverse=True)


def count_grid_votes(
    raw_detections: list[RawDetection],
    grid_size: int,
) -> dict[tuple[int, int], int]:
    votes: dict[tuple[int, int], int] = {}
    for raw in raw_detections:
        center_x = (raw.x1 + raw.x2) / 2.0
        center_y = (raw.y1 + raw.y2) / 2.0
        cell = (int(center_y // grid_size), int(center_x // grid_size))
        votes[cell] = votes.get(cell, 0) + 1
    return votes


def apply_tile_vote_boost(
    detections: list[Detection],
    vote_counts: dict[tuple[int, int], int],
    grid_size: int,
) -> list[Detection]:
    boosted: list[Detection] = []
    for detection in detections:
        cell = grid_cell_key(detection.bbox, grid_size)
        votes = vote_counts.get(cell, 1)
        multiplier = 1.0 + 0.1 * min(max(votes - 1, 0), 10)
        boosted.append(
            Detection(
                class_id=detection.class_id,
                label=detection.label,
                confidence=min(detection.confidence * multiplier, 1.0),
                bbox=detection.bbox,
            )
        )
    return boosted


def refine_wally_detections(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    *,
    grid_size: int,
    snap_to_grid: bool,
    cluster_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
    vote_counts: dict[tuple[int, int], int] | None = None,
) -> list[Detection]:
    if not detections:
        return []

    merged = cluster_merge_detections(detections, iou_threshold=cluster_iou)
    sized = filter_by_box_area(merged, grid_size, min_area_ratio, max_area_ratio)
    if vote_counts:
        sized = apply_tile_vote_boost(sized, vote_counts, grid_size)
    if snap_to_grid:
        sized = snap_detections_to_grid(sized, grid_size, image_width, image_height)
        deduped = dedupe_grid_cells(sized, grid_size)
    else:
        deduped = sized

    if vote_counts:
        return sorted(
            deduped,
            key=lambda item: (
                vote_counts.get(grid_cell_key(item.bbox, grid_size), 0),
                item.confidence,
            ),
            reverse=True,
        )
    return sorted(deduped, key=lambda item: item.confidence, reverse=True)


def raw_to_detections_after_nms(
    raw: list[RawDetection],
    merge_iou: float,
) -> list[Detection]:
    from wally_ai_api.infrastructure.model.prediction_mapper import map_raw_detections

    filtered = nms_detections(raw, iou_threshold=merge_iou)
    return map_raw_detections(filtered)
