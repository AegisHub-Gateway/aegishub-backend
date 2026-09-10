"""
Timothy's Responsibility — Sign Language Stream Schemas
========================================================
Pydantic v2 models validating continuous time-series 3D hand-landmark
data (e.g. captured client-side via MediaPipe Hands) before it is handed
off to Scroll's LSTM sign-classification model.

Any structurally invalid payload — an empty/missing frame array, a frame
with the wrong landmark count, or a coordinate outside the normalized
[0.0, 1.0] range — fails validation here and FastAPI automatically
surfaces it as an HTTP 422 Unprocessable Entity, before any inference
code ever runs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# MediaPipe Hands emits exactly 21 landmarks per detected hand.
LANDMARKS_PER_FRAME = 21

# The LSTM model expects a fixed-length temporal window of frames.
EXPECTED_FRAME_COUNT = 30

# Coordinates are normalized relative to image width/height/depth.
COORD_MIN = 0.0
COORD_MAX = 1.0


class Landmark(BaseModel):
    """A single normalized 3D hand-landmark coordinate."""

    x: float = Field(..., ge=COORD_MIN, le=COORD_MAX, description="Normalized X coordinate.")
    y: float = Field(..., ge=COORD_MIN, le=COORD_MAX, description="Normalized Y coordinate.")
    z: float = Field(..., ge=COORD_MIN, le=COORD_MAX, description="Normalized relative depth.")


class HandFrame(BaseModel):
    """A single video frame, containing exactly 21 hand landmarks."""

    landmarks: List[Landmark] = Field(
        ...,
        min_length=LANDMARKS_PER_FRAME,
        max_length=LANDMARKS_PER_FRAME,
        description=f"Exactly {LANDMARKS_PER_FRAME} hand landmarks for this frame.",
    )

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(cls, value: List[Landmark]) -> List[Landmark]:
        if len(value) != LANDMARKS_PER_FRAME:
            raise ValueError(
                f"Each frame must contain exactly {LANDMARKS_PER_FRAME} landmarks, got {len(value)}."
            )
        return value


class SignStreamRequest(BaseModel):
    """
    A continuous time-series window of hand-landmark frames, ready for
    LSTM inference.
    """

    frames: List[HandFrame] = Field(
        ...,
        min_length=1,
        description=f"Sequence of hand-landmark frames (expected length: {EXPECTED_FRAME_COUNT}).",
    )
    session_id: Optional[str] = Field(
        default=None, description="Optional client-generated session/stream identifier."
    )

    @field_validator("frames")
    @classmethod
    def validate_frames_not_empty(cls, value: List[HandFrame]) -> List[HandFrame]:
        if not value:
            raise ValueError("`frames` must not be an empty array.")
        return value

    @model_validator(mode="after")
    def validate_frame_sequence_length(self) -> "SignStreamRequest":
        if len(self.frames) != EXPECTED_FRAME_COUNT:
            raise ValueError(
                f"Expected exactly {EXPECTED_FRAME_COUNT} frames for LSTM inference, "
                f"got {len(self.frames)}."
            )
        return self


class SignPredictionResponse(BaseModel):
    """Response returned after running the LSTM sign-classification model."""

    detected_sign: str = Field(..., description="Predicted sign/gesture label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    is_emergency: bool = Field(..., description="True if the detected sign maps to an emergency phrase.")
