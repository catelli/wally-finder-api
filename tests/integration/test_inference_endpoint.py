import base64

import pytest
from fastapi.testclient import TestClient

from tests.conftest import INVALID_FILE, VALID_IMAGE


@pytest.mark.integration
def test_inference_json_returns_annotated_image(client: TestClient) -> None:
    with VALID_IMAGE.open("rb") as handle:
        response = client.post(
            "/api/v1/inference",
            files={"file": ("scene.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "request_id" in payload
    assert "detections" in payload
    assert "annotated_image" in payload
    assert payload["annotated_image"]["content_type"] == "image/jpeg"
    decoded = base64.b64decode(payload["annotated_image"]["data_base64"])
    assert len(decoded) > 1000


@pytest.mark.integration
def test_inference_image_format_returns_jpeg(client: TestClient) -> None:
    with VALID_IMAGE.open("rb") as handle:
        response = client.post(
            "/api/v1/inference?response_format=image",
            files={"file": ("scene.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "X-Request-Id" in response.headers
    assert len(response.content) > 1000


@pytest.mark.integration
def test_inference_rejects_invalid_file(client: TestClient) -> None:
    with INVALID_FILE.open("rb") as handle:
        response = client.post(
            "/api/v1/inference",
            files={"file": ("invalid.txt", handle, "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_media_type"
