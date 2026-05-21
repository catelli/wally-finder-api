from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wally_ai_api.app.main import create_app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_IMAGE = FIXTURES_DIR / "images" / "valid_scene.jpg"
INVALID_FILE = FIXTURES_DIR / "images" / "invalid_file.txt"


@pytest.fixture(scope="session")
def valid_image_bytes() -> bytes:
    return VALID_IMAGE.read_bytes()


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client
