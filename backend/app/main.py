"""FastAPI application entrypoint.

Run locally with::

    uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port 8000 --reload

Lifespan order:
    1. Load YAML ``AppConfig`` (research-side hyperparameters).
    2. Load weights + tokenizer into a ``CaptionPredictor`` singleton.
    3. Optionally warmup so the first request doesn't pay TF's lazy build cost.
    4. Wrap the predictor in a ``PredictorService`` and stash on app state.

The singleton lives on ``app.state.predictor_service``; routes pull it
through a ``Depends`` so tests can override the dependency cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import BackendSettings, get_backend_settings
from app.core.logging import RequestContextMiddleware, configure_app_logging
from app.services.predictor_service import PredictorService
from captioning.config import load_config
from captioning.config.schema import AppConfig
from captioning.inference import CaptionPredictor
from captioning.utils import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the predictor at startup, release it at shutdown."""
    settings: BackendSettings = app.state.backend_settings
    config: AppConfig = app.state.app_config

    log.info(
        "predictor_loading",
        weights=str(settings.weights_path),
        tokenizer_dir=str(settings.tokenizer_dir),
        model_version=settings.model_version,
    )

    predictor = CaptionPredictor.from_artifacts(
        weights_path=settings.weights_path,
        tokenizer_dir=settings.tokenizer_dir,
        config=config,
    )
    if settings.warmup:
        predictor.warmup()

    app.state.predictor_service = PredictorService(
        predictor=predictor,
        model_version=settings.model_version,
        max_upload_bytes=config.serve.max_upload_bytes,
    )
    log.info("predictor_ready", model_version=settings.model_version)

    try:
        yield
    finally:
        app.state.predictor_service = None
        log.info("predictor_unloaded")


def create_app() -> FastAPI:
    """Build the FastAPI app. Factory form so tests can construct fresh apps."""
    configure_app_logging()
    settings = get_backend_settings()
    config = load_config(settings.config_path)

    app = FastAPI(
        title="Image Captioning API",
        version=settings.api_version,
        description=(
            "Production-grade inference service for the IEEE-published "
            "CNN+Transformer image captioning model."
        ),
        lifespan=lifespan,
    )

    app.state.backend_settings = settings
    app.state.app_config = config
    app.state.predictor_service = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.serve.cors_allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(router)
    return app


app = create_app()
