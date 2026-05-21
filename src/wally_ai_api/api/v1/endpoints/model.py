from fastapi import APIRouter, Depends

from wally_ai_api.api.v1.schemas.model import ModelMetadataResponseSchema
from wally_ai_api.app.dependencies import get_model_metadata_use_case
from wally_ai_api.application.use_cases.get_model_metadata import GetModelMetadataUseCase

router = APIRouter(prefix="/model", tags=["model"])


@router.get("")
def get_model_metadata(
    use_case: GetModelMetadataUseCase = Depends(get_model_metadata_use_case),
) -> ModelMetadataResponseSchema:
    metadata = use_case.execute()
    return ModelMetadataResponseSchema(
        name=metadata.name,
        version=metadata.version,
        classes=metadata.classes,
        input_size=metadata.input_size,
        tiled_inference=metadata.tiled_inference,
    )
