from pydantic import BaseModel


class HealthResponseSchema(BaseModel):
    status: str
    model_loaded: bool
