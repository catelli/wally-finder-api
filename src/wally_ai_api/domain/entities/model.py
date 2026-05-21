from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    classes: list[str]
    input_size: int
    tiled_inference: bool
