"""
Neche's Responsibility — Derma-Scan Router
===========================================
Accepts a skin-lesion image upload, preprocesses it with OpenCV/NumPy,
and routes it through Scroll's MobileNet derma-classification model
(currently a dummy fallback — see `app/models/model_loader.py`).
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.model_loader import get_model_registry
from app.schemas.derma import DermaScanResponse
from app.services.preprocessing import process_derma_image

logger = logging.getLogger("aegishub.routers.derma")

router = APIRouter(prefix="/derma", tags=["Derma-Scan"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
TARGET_SIZE = (224, 224)  # MobileNet input resolution

URGENCY_SUMMARIES = {
    "low": "No urgent action needed. Consider monitoring for changes over the coming weeks.",
    "medium": "Recommend scheduling a dermatologist consultation within the next 1-2 weeks.",
    "high": "Please seek in-person dermatological evaluation as soon as possible.",
}


@router.post(
    "/scan",
    response_model=DermaScanResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a skin-lesion image for urgency triage",
)
async def scan_derma_image(
    image: UploadFile = File(..., description="Skin lesion photo (JPEG/PNG/WebP)."),
) -> DermaScanResponse:
    """
    Accepts a multipart image upload, preprocesses it with OpenCV, and
    returns a MobileNet classification (dummy fallback until real weights
    are supplied) plus a patient-facing urgency summary.
    """
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported content type '{image.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    raw_bytes = await image.read()

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty."
        )

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    # Process image using service function
    try:
        image_tensor = process_derma_image(raw_bytes, target_size=TARGET_SIZE)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    registry = get_model_registry()
    result = registry.predict_derma(image_tensor)
    urgency = result["urgency"]

    # Add processed image shape to response
    result["processed_image_shape"] = list(image_tensor.shape)

    logger.info(
        "Derma scan | filename=%s | condition=%s | urgency=%s | confidence=%.4f | image_shape=%s",
        image.filename,
        result["condition"],
        urgency,
        result["confidence"],
        result["processed_image_shape"],
    )

    return DermaScanResponse(
        condition=result["condition"],
        urgency=urgency,
        confidence=result["confidence"],
        summary=URGENCY_SUMMARIES.get(
            urgency, "Please consult a medical professional for further evaluation."
        ),
        processed_image_shape=result["processed_image_shape"],
    )
