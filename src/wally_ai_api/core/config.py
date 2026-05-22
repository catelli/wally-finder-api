from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml_config(name: str) -> dict[str, Any]:
    path = project_root() / "configs" / name
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WALLY_API_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_bytes: int = 20_971_520
    allowed_content_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WALLY_MODEL_", extra="ignore")

    weights_path: str = "models/wally_tiles_best.pt"
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.45
    device: str = "0"
    imgsz: int = 256
    tile_size: int = 256
    tile_overlap: int = 64
    merge_iou: float = 0.45
    max_output_detections: int = 4
    min_output_confidence: float = 0.88
    warmup_on_startup: bool = True
    class_names: list[str] = Field(default_factory=lambda: ["wally"])


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WALLY_LOG_", extra="ignore")

    level: str = "INFO"
    json_logs: bool = True


@lru_cache
def get_app_settings() -> AppSettings:
    yaml_data = load_yaml_config("app.yaml")
    return AppSettings(**yaml_data)


@lru_cache
def get_model_settings() -> ModelSettings:
    yaml_data = load_yaml_config("model.yaml")
    return ModelSettings(**yaml_data)


@lru_cache
def get_logging_settings() -> LoggingSettings:
    yaml_data = load_yaml_config("logging.yaml")
    return LoggingSettings(**yaml_data)


def resolve_weights_path(settings: ModelSettings | None = None) -> Path:
    model_settings = settings or get_model_settings()
    weights = Path(model_settings.weights_path)
    if weights.is_absolute():
        return weights
    return project_root() / weights
