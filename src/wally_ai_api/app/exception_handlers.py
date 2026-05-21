from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from wally_ai_api.api.v1.schemas.errors import ErrorResponseSchema
from wally_ai_api.domain.exceptions.image_errors import ImageValidationError
from wally_ai_api.domain.exceptions.inference_errors import InferenceError
from wally_ai_api.domain.exceptions.model_errors import ModelNotLoadedError
from wally_ai_api.utils.id_utils import new_request_id


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ImageValidationError)
    async def image_validation_handler(
        _request: Request,
        exc: ImageValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponseSchema(
                code=exc.code,
                message=exc.message,
                request_id=new_request_id(),
            ).model_dump(),
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(
        _request: Request,
        exc: ModelNotLoadedError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=ErrorResponseSchema(
                code=exc.code,
                message=exc.message,
                request_id=new_request_id(),
            ).model_dump(),
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        _request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponseSchema(
                code=exc.code,
                message=exc.message,
                request_id=new_request_id(),
            ).model_dump(),
        )
