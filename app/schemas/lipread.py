"""
Neche's Responsibility — Lip-Reading Schemas
=============================================
Pydantic v2 models for the audio-visual lip-reading endpoint. Audio
streams in as a multipart `UploadFile`; the lip-mesh coordinates arrive
as a JSON-encoded string form field (`lip_mesh`) and are parsed and
validated against `LipMeshPayload` in the router before being handed to
the inference pipeline.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

# Facial lip-contour landmark sets vary in density by client-side model
# (e.g. MediaPipe FaceMesh's lip subset), so a permissive minimum is
# enforced rather than an exact count.
MIN_LIP_POINTS = 4

COORD_MIN = 0.0
COORD_MAX = 1.0


class LipMeshPoint(BaseModel):
    """A single normalized lip-contour coordinate."""

    x: float = Field(..., ge=COORD_MIN, le=COORD_MAX)
    y: float = Field(..., ge=COORD_MIN, le=COORD_MAX)
    z: float = Field(default=0.0, ge=COORD_MIN, le=COORD_MAX, description="Optional relative depth.")


class LipMeshFrame(BaseModel):
    """A single frame of lip-mesh coordinates, synchronized to the audio track."""

    points: List[LipMeshPoint] = Field(..., min_length=MIN_LIP_POINTS)

    @field_validator("points")
    @classmethod
    def validate_points_not_empty(cls, value: List[LipMeshPoint]) -> List[LipMeshPoint]:
        if len(value) < MIN_LIP_POINTS:
            raise ValueError(f"Each lip-mesh frame requires at least {MIN_LIP_POINTS} points.")
        return value


class LipMeshPayload(BaseModel):
    """The full parsed `lip_mesh` JSON payload submitted alongside the audio file."""

    frames: List[LipMeshFrame] = Field(..., min_length=1)

    @field_validator("frames")
    @classmethod
    def validate_frames_not_empty(cls, value: List[LipMeshFrame]) -> List[LipMeshFrame]:
        if not value:
            raise ValueError("`frames` must not be an empty array.")
        return value


class LipReadResponse(BaseModel):
    """Synchronized subtitle result returned after audio-visual inference."""

    transcript: str = Field(..., description="Predicted spoken transcript.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    is_muffled: bool = Field(
        ..., description="True when audio quality is low and lip-reading was the primary signal."
    )
