"""AegisHub Sign Language API application entrypoint."""

from fastapi import FastAPI

from app.config import settings
from app.schemas import HealthResponse, SignInput, SignPredictionResponse
from app.sign_service import get_sign_service

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.MODEL_VERSION,
)


@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health and model readiness check",
)
def healthz() -> HealthResponse:
    """Health check endpoint indicating service health and model status."""
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        model_version=settings.MODEL_VERSION,
        model_loaded=settings.MODEL_LOADED,
    )


@app.post(
    "/v1/sign/classify",
    response_model=SignPredictionResponse,
    tags=["Sign Classification"],
    summary="Classify sign language gesture from landmark stream",
)
def classify_sign(payload: SignInput) -> SignPredictionResponse:
    """Accepts a temporal sequence of hand-landmark frames and returns predicted sign."""
    service = get_sign_service()
    result = service.predict_sign(payload)
    return SignPredictionResponse(**result)


# --- Aliases for convenience ---
@app.post("/api/v1/sign/classify", response_model=SignPredictionResponse, include_in_schema=False)
def classify_sign_alias(payload: SignInput) -> SignPredictionResponse:
    """Alias for /v1/sign/classify."""
    return classify_sign(payload)


@app.get("/health", response_model=HealthResponse, tags=["Health"], include_in_schema=False)
def health() -> HealthResponse:
    """Alias for /healthz."""
    return healthz()


@app.get("/", response_model=HealthResponse, tags=["Meta"], include_in_schema=False)
def root() -> HealthResponse:
    """Root endpoint alias returning service health metadata."""
    return healthz()
