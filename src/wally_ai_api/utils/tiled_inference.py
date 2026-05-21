from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileWindow:
    x_offset: int
    y_offset: int
    width: int
    height: int


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def compute_tile_windows(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int = 0,
) -> list[TileWindow]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    stride = tile_size - overlap
    windows: list[TileWindow] = []

    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            width = min(tile_size, image_width - x)
            height = min(tile_size, image_height - y)
            if width < tile_size // 4 or height < tile_size // 4:
                continue
            windows.append(TileWindow(x_offset=x, y_offset=y, width=width, height=height))

    return windows


def extract_tile(image: Image.Image, window: TileWindow, tile_size: int) -> Image.Image:
    tile = image.crop(
        (
            window.x_offset,
            window.y_offset,
            window.x_offset + window.width,
            window.y_offset + window.height,
        )
    )
    if tile.size != (tile_size, tile_size):
        canvas = Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))
        canvas.paste(tile, (0, 0))
        return canvas
    return tile


def shift_boxes_to_global(boxes_xyxy: np.ndarray, window: TileWindow) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4)
    shifted = boxes_xyxy.copy().astype(float)
    shifted[:, [0, 2]] += window.x_offset
    shifted[:, [1, 3]] += window.y_offset
    return shifted


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_detections(
    detections: list[RawDetection],
    iou_threshold: float = 0.45,
) -> list[RawDetection]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[RawDetection] = []

    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        best_box = np.array([best.x1, best.y1, best.x2, best.y2])
        ordered = [
            candidate
            for candidate in ordered
            if iou(
                best_box,
                np.array([candidate.x1, candidate.y1, candidate.x2, candidate.y2]),
            )
            < iou_threshold
        ]

    return kept
