from wally_ai_api.domain.entities.model import ModelMetadata
from wally_ai_api.domain.exceptions.model_errors import ModelNotLoadedError
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort


class GetModelMetadataUseCase:
    def __init__(self, inference_engine: InferenceEnginePort) -> None:
        self._inference_engine = inference_engine

    def execute(self) -> ModelMetadata:
        if not self._inference_engine.is_loaded():
            raise ModelNotLoadedError()
        return self._inference_engine.metadata()
