from wally_ai_api.application.use_cases.get_model_metadata import GetModelMetadataUseCase
from wally_ai_api.domain.entities.model import ModelMetadata
from wally_ai_api.domain.ports.inference_engine import InferenceEnginePort


class StubEngine(InferenceEnginePort):
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def predict(self, image):  # noqa: ANN001, ARG002
        return []

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="wally_tiles",
            version="test",
            classes=["wally"],
            input_size=256,
            tiled_inference=True,
        )


def test_get_model_metadata_use_case() -> None:
    use_case = GetModelMetadataUseCase(StubEngine())
    metadata = use_case.execute()
    assert metadata.tiled_inference is True
    assert metadata.classes == ["wally"]
