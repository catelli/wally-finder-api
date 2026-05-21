import uvicorn

from wally_ai_api.core.config import get_app_settings


def main() -> None:
    settings = get_app_settings()
    uvicorn.run(
        "wally_ai_api.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
