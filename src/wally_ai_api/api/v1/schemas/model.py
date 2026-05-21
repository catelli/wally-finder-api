from pydantic import BaseModel


class ModelMetadataResponseSchema(BaseModel):
    name: str
    version: str
    classes: list[str]
    input_size: int
    tiled_inference: bool
