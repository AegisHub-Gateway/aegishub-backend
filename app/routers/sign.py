"""
Timothy's Responsibility — Sign Language Stream Router
=======================================================
Accepts a continuous time-series window of hand-landmark frames and
routes it through Scroll's LSTM sign-classification model (currently a
dummy fallback — see `app/models/model_loader.py`).
"""

import logging

import numpy as np
from fastapi import APIRouter, status

from app.models.model_loader import get_model_registry
from app.schemas.sign import SignPredictionResponse, SignStreamRequest

logger = logging.getLogger("aegishub.routers.sign")

router = APIRouter(prefix="/sign", tags=["Sign Language"])


def _landmarks_to_tensor(payload: SignStreamRequest) -> np.ndarray:
    """Flatten a validated `SignStreamRequest` into a `(frames, 21, 3)` float32 array."""
    return np.array(
        [[[lm.x, lm.y, lm.z] for lm in frame.landmarks] for frame in payload.frames],
        dtype=np.float32,
    )


@router.post(
    "/predict",
    response_model=SignPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a sign-language gesture from a hand-landmark time series",
)
async def predict_sign(payload: SignStreamRequest) -> SignPredictionResponse:
    """
    Accepts a `SignStreamRequest` — a fixed-length sequence of 21-point
    hand-landmark frames — and returns the LSTM model's predicted sign,
    confidence score, and whether the sign maps to an emergency phrase.

    Any malformed payload (missing frames, wrong landmark count, or
    coordinates outside the normalized [0.0, 1.0] range) never reaches
    this function body — Pydantic rejects it beforehand with an
    automatic HTTP 422 response.
    """
    landmark_tensor = _landmarks_to_tensor(payload)
    registry = get_model_registry()
    result = registry.predict_sign(landmark_tensor)

    logger.info(
        "Sign prediction | session=%s | sign=%s | confidence=%.4f | emergency=%s",
        payload.session_id,
        result["detected_sign"],
        result["confidence"],
        result["is_emergency"],
    )

    return SignPredictionResponse(**result)
