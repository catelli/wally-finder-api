from fastapi import APIRouter, Request

from wally_ai_api.api.v1.schemas.health import HealthResponseSchema

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check(request: Request) -> HealthResponseSchema:
    engine = request.app.state.inference_engine
    return HealthResponseSchema(
        status="ok",
        model_loaded=engine.is_loaded(),
    )
