from fastapi import Request

from wally_ai_api.application.use_cases.detect_wally import DetectWallyUseCase
from wally_ai_api.application.use_cases.get_model_metadata import GetModelMetadataUseCase


def get_detect_wally_use_case(request: Request) -> DetectWallyUseCase:
    return request.app.state.detect_wally_use_case


def get_model_metadata_use_case(request: Request) -> GetModelMetadataUseCase:
    return request.app.state.get_model_metadata_use_case
