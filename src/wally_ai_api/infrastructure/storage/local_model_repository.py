from pathlib import Path

from wally_ai_api.core.config import resolve_weights_path
from wally_ai_api.domain.ports.model_repository import ModelRepositoryPort


class LocalModelRepository(ModelRepositoryPort):
    def __init__(self, weights_path: Path | None = None) -> None:
        self._weights_path = weights_path or resolve_weights_path()

    def get_weights_path(self) -> Path:
        return self._weights_path

    def weights_exist(self) -> bool:
        return self._weights_path.is_file()
