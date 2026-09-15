"""
Neche's Responsibility — Lip-Reading Router
============================================
Accepts synchronized audio + facial lip-mesh coordinates and routes them
through the audio-visual lip-reading inference pipeline (currently a
dummy fallback — see `app/models/model_loader.py`).
"""

import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.models.model_loader import get_model_registry
from app.schemas.lipread import LipMeshPayload, LipReadResponse
from app.services.preprocessing import process_audio_file, process_lip_landmarks

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
    audio: UploadFile = File(
        ..., description="Audio clip synchronized with the lip-mesh frames."
    ),
    lip_mesh: str = Form(
        ..., description="JSON-encoded array of lip-mesh coordinate frames."
    ),
) -> LipReadResponse:
    """
    Accepts a multipart request containing an audio clip and a JSON-encoded
    `lip_mesh` coordinate string, cross-references both signals, and
    returns a synchronized subtitle prediction.
    """
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported audio content type '{audio.content_type}'. "
                f"Allowed: {sorted(ALLOWED_AUDIO_TYPES)}."
            ),
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded audio file is empty.",
        )

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Uploaded audio exceeds the "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."
            ),
        )

    # 1. Parse and validate JSON lip_mesh
    try:
        mesh_data = json.loads(lip_mesh)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"`lip_mesh` is not valid JSON: {exc.msg}",
        ) from exc

    # Validate against schema
    mesh_payload = _parse_lip_mesh(lip_mesh)

    # 2. Preprocess lip landmark sequence
    try:
        landmark_array = process_lip_landmarks(mesh_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process lip landmarks: {str(exc)}",
        ) from exc

    # 3. Save audio temporarily & extract MFCC audio features
    tmp_path = None
    mfcc_features = None
    try:
        suffix = os.path.splitext(audio.filename or "audio")[1] or ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Process audio file into MFCC array
        mfcc_features = process_audio_file(tmp_path, sr=16000, n_mfcc=13)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process audio file: {str(exc)}",
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4. Run inference and return response
    registry = get_model_registry()
    result = registry.dummy_predict_lipread(
        transcript_seed=audio_bytes,
        lip_mesh_frame_count=len(mesh_payload.frames),
    )

    # Add processed frames count and audio features shape to response
    result["processed_frames_count"] = len(landmark_array)
    result["audio_mfcc_shape"] = (
        list(mfcc_features.shape) if mfcc_features is not None else []
    )

    logger.info(
        (
            "Lip-read transcription | filename=%s | frames=%d | "
            "confidence=%.4f | muffled=%s | processed_frames=%d | mfcc_shape=%s"
        ),
        audio.filename,
        len(mesh_payload.frames),
        result["confidence"],
        result["is_muffled"],
        len(landmark_array),
        result["audio_mfcc_shape"],
    )

    return LipReadResponse(**result)
