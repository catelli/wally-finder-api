from pydantic import BaseModel


class ErrorResponseSchema(BaseModel):
    code: str
    message: str
    request_id: str | None = None
