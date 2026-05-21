from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from wally_ai_api.application.use_cases.detect_wally import DetectWallyUseCase
from wally_ai_api.application.use_cases.get_model_metadata import GetModelMetadataUseCase
from wally_ai_api.core.config import get_model_settings
from wally_ai_api.core.logging import get_logger, setup_logging
from wally_ai_api.infrastructure.image.opencv_image_processor import OpenCvImageProcessor
from wally_ai_api.infrastructure.model.yolo_inference_engine import YoloTiledInferenceEngine
from wally_ai_api.infrastructure.storage.local_model_repository import LocalModelRepository

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = get_model_settings()
    repository = LocalModelRepository()
    inference_engine = YoloTiledInferenceEngine(repository)
    image_processor = OpenCvImageProcessor()

    logger.info("loading_model", weights=str(repository.get_weights_path()))
    inference_engine.load()
    if settings.warmup_on_startup:
        inference_engine.warmup()
        logger.info("model_warmup_complete")

    app.state.inference_engine = inference_engine
    app.state.image_processor = image_processor
    app.state.detect_wally_use_case = DetectWallyUseCase(inference_engine, image_processor)
    app.state.get_model_metadata_use_case = GetModelMetadataUseCase(inference_engine)

    yield

    logger.info("application_shutdown")
