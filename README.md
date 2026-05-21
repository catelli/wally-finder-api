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
set WALLY_MODEL_DEVICE=cpu
python scripts/run_api.py
```

Weights: copy from training `runs/detect/runs/wally_tiles_train/weights/best.pt` to `models/wally_tiles_best.pt` (already done in this repo).

API: http://localhost:8000/docs

## Public URL (ngrok)

Reserved domain: **https://wallyfinder-api.ngrok.app**

1. Install ngrok (once) — binary is saved under `%LOCALAPPDATA%\ngrok\` if you used the project setup.
2. Configure token (once, do not commit the token):

```powershell
ngrok config add-authtoken YOUR_TOKEN
```

3. Start the API on port `8000`, then expose it:

```powershell
# Terminal 1
python scripts/run_api.py

# Terminal 2
.\scripts\start_ngrok.ps1
```

Or manually:

```powershell
ngrok http 8000 --url https://wallyfinder-api.ngrok.app
```

Health check: `https://wallyfinder-api.ngrok.app/api/v1/health`

Inference from the front:

```text
POST https://wallyfinder-api.ngrok.app/api/v1/inference
```

## Inference (front)

`POST /api/v1/inference` with multipart field `file`.

**JSON (default)** — annotated image as base64 plus boxes:

```bash
curl -X POST "https://wallyfinder-api.ngrok.app/api/v1/inference" -F "file=@scene.jpg"
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