from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wally_ai_api.api.router import router as api_router
from wally_ai_api.app.exception_handlers import register_exception_handlers
from wally_ai_api.app.lifespan import lifespan
from wally_ai_api.core.config import get_app_settings


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(
        title="Wally AI API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
