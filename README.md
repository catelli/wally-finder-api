# wally-finder-api

Production FastAPI service for Wally detection using the **tiles** YOLO model.

## Model

Weights live at `models/wally_tiles_best.pt` (copied from tile training `best.pt`).

## Run locally

```bash
cd wally-finder-api
# Reuse wally-ai-training venv (already has torch/ultralytics) or create a new venv:
..\wally-ai-training\.venv\Scripts\activate
pip install -e ".[dev]"
set WALLY_MODEL_DEVICE=0
python scripts/run_api.py
```

Inference uses **GPU** (`device: 0` in `configs/model.yaml`). Override with `WALLY_MODEL_DEVICE=cpu` only for debugging.

Weights: copy from training `runs/detect/runs/wally_tiles_train/weights/best.pt` to `models/wally_tiles_best.pt` (already done in this repo).

API: http://localhost:8000/docs

Health: `GET http://localhost:8000/api/v1/health`

## Inference (front)

`POST /api/v1/inference` with multipart field `file`.

**JSON (default)** — annotated image as base64 plus boxes:

```bash
curl -X POST "http://localhost:8000/api/v1/inference" -F "file=@scene.jpg"
```

Response fields: `request_id`, `detections[]`, `annotated_image.data_base64`.

**Raw JPEG** — for `<img src>` or blob display:

```bash
curl -X POST "http://localhost:8000/api/v1/inference?response_format=image" -F "file=@scene.jpg" --output annotated.jpg
```

Headers: `X-Request-Id`, `X-Detection-Count`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Integration tests load the real model (CPU).

## Cursor rules

`.cursor/rules/wally-ai-api-architecture.mdc`