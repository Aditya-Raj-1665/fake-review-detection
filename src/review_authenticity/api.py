"""Small stateless FastAPI application with one model per worker."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from review_authenticity.config import Settings
from review_authenticity.inference import InferenceService
from review_authenticity.logging_config import configure_logging

logger = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
    """Accept exactly one review string."""

    model_config = ConfigDict(extra="forbid")
    text: StrictStr


class BatchRequest(BaseModel):
    """Accept a bounded list of review strings."""

    model_config = ConfigDict(extra="forbid")
    reviews: list[StrictStr] = Field(min_length=1, max_length=128)


class ModelSignal(BaseModel):
    """Describe a statistical feature contribution, not evidence of deception."""

    term: str
    direction: Literal["genuine", "deceptive"]
    weight: float


class Prediction(BaseModel):
    """Publish a probability-based decision and its model provenance."""

    label: Literal["genuine", "deceptive"]
    authenticity_score: float = Field(ge=0, le=100)
    genuine_probability: float = Field(ge=0, le=1)
    deceptive_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    model_name: str
    model_version: str
    demo: bool
    threshold: float
    inference_ms: float
    model_signals: list[ModelSignal]


class ModelInfo(BaseModel):
    """Describe the active model and input limits."""

    model_name: str
    model_version: str
    model_type: str
    demo: bool
    threshold: float
    calibration: str
    max_review_length: int
    max_batch_size: int
    max_tokens: int | None


class BodyLimitMiddleware:
    """Bound request bytes even when Content-Length is absent or dishonest."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                response = JSONResponse({"detail": "Request body is too large."}, status_code=413)
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            buffered = next(iterator, None)
            return buffered if buffered is not None else await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application whose lifespan owns its model and configuration."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        logger.info("Application starting")
        app.state.inference = InferenceService(settings)
        yield
        app.state.inference = None
        logger.info("Application stopped")

    app = FastAPI(title="Review Authenticity API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        """Return useful validation details without echoing submitted text."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
            },
        )

    def get_service() -> InferenceService:
        service = getattr(app.state, "inference", None)
        if service is None:
            raise HTTPException(503, "Model is unavailable.")
        return service

    def validate_texts(texts: list[str]) -> None:
        if len(texts) > settings.max_batch_size:
            raise HTTPException(422, f"At most {settings.max_batch_size} reviews per request.")
        for index, text in enumerate(texts):
            if not text.strip():
                raise HTTPException(422, f"Review {index + 1} must not be empty.")
            if len(text) > settings.max_review_length:
                raise HTTPException(
                    422, f"Review {index + 1} exceeds {settings.max_review_length} characters."
                )

    def run_prediction(texts: list[str]) -> list[dict]:
        validate_texts(texts)
        service = get_service()
        try:
            return service.predict(texts)
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(500, "Prediction failed. Please try again.") from None

    @app.get("/health")
    def health() -> dict:
        """Report readiness only when a model has loaded."""
        get_service()
        return {"status": "healthy", "model_loaded": True}

    @app.get("/api/v1/model/info", response_model=ModelInfo)
    def model_info():
        """Expose active model details and configured input limits."""
        return get_service().info()

    @app.post("/api/v1/predict", response_model=Prediction)
    def predict(payload: ReviewRequest):
        """Analyze one review in FastAPI's worker thread pool."""
        return run_prediction([payload.text])[0]

    @app.post("/api/v1/predict/batch", response_model=list[Prediction])
    def predict_batch(payload: BatchRequest):
        """Analyze a bounded batch with one model call."""
        return run_prediction(payload.reviews)

    return app
