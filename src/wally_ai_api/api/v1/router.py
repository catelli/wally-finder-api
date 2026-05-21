from fastapi import APIRouter

from wally_ai_api.api.v1.endpoints import health, inference, model

router = APIRouter()
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(model.router)
