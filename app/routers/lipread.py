"""
Neche's Responsibility — Lip-Reading Router
============================================
Accepts synchronized audio + facial lip-mesh coordinates and routes them
through the audio-visual lip-reading inference pipeline (currently a
dummy fallback — see `app/models/model_loader.py`).
"""

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.models.model_loader import get_model_registry
from app.schemas.lipread import LipMeshPayload, LipReadResponse

logger = logging.getLogger("aegishub.routers.lipread")

router = APIRouter(prefix="/lipread", tags=["Lip-Reading"])

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/webm", "audio/ogg",
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def _parse_lip_mesh(lip_mesh: str) -> LipMeshPayload:
    """Parse and validate the `lip_mesh` JSON-string form field."""
    try:
        raw = json.loads(lip_mesh)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"`lip_mesh` is not valid JSON: {exc.msg}",
        ) from exc

    try:
        return LipMeshPayload.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=json.loads(exc.json()),
        ) from exc


@router.post(
    "/transcribe",
    response_model=LipReadResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate synchronized subtitles from audio + lip-mesh coordinates",
)
async def transcribe_lipread(
    audio: UploadFile = File(..., description="Audio clip synchronized with the lip-mesh frames."),
    lip_mesh: str = Form(..., description="JSON-encoded array of lip-mesh coordinate frames."),
) -> LipReadResponse:
    """
    Accepts a multipart request containing an audio clip and a JSON-encoded
    `lip_mesh` coordinate string, cross-references both signals, and
    returns a synchronized subtitle prediction.
    """
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio content type '{audio.content_type}'. Allowed: {sorted(ALLOWED_AUDIO_TYPES)}.",
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded audio file is empty."
        )

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded audio exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    mesh_payload = _parse_lip_mesh(lip_mesh)

    registry = get_model_registry()
    result = registry.dummy_predict_lipread(
        transcript_seed=audio_bytes,
        lip_mesh_frame_count=len(mesh_payload.frames),
    )

    logger.info(
        "Lip-read transcription | filename=%s | frames=%d | confidence=%.4f | muffled=%s",
        audio.filename,
        len(mesh_payload.frames),
        result["confidence"],
        result["is_muffled"],
    )

    return LipReadResponse(**result)
