"""Pydantic schemas for the AegisHub Sign Language API.

Defines the landmark contract, health status, and prediction schemas.
"""

from pydantic import BaseModel, Field

# Coordinate normalization bounds for planar (x, y) coordinates
COORD_MIN: float = 0.0
COORD_MAX: float = 1.0


class Landmark(BaseModel):
    """A single landmark point with normalized (x, y) and relative depth z."""

    x: float = Field(
        ...,
        ge=COORD_MIN,
        le=COORD_MAX,
        description="Normalized horizontal coordinate in [0.0, 1.0].",
    )
    y: float = Field(
        ...,
        ge=COORD_MIN,
        le=COORD_MAX,
        description="Normalized vertical coordinate in [0.0, 1.0].",
    )
    z: float = Field(
        ...,
        description="Relative depth coordinate (unconstrained scale; accepts negative values).",
    )


class Frame(BaseModel):
    """A single temporal frame capturing left and/or right hand landmarks."""

    left_hand: list[Landmark] = Field(
        default_factory=list,
        description="List of landmarks for the left hand; empty if not detected.",
    )
    right_hand: list[Landmark] = Field(
        default_factory=list,
        description="List of landmarks for the right hand; empty if not detected.",
    )


class SignInput(BaseModel):
    """Input payload containing a sequence of landmark frames."""

    frames: list[Frame] = Field(
        default_factory=list,
        description="Ordered sequence of temporal landmark frames.",
    )


class HealthResponse(BaseModel):
    """Schema representing service health and model readiness status."""

    status: str = Field(..., description="Service status indicator, e.g. 'ok'.")
    service: str = Field(..., description="Registered service name.")
    model_version: str = Field(..., description="Identifier tag of the current model.")
    model_loaded: bool = Field(..., description="Indicates whether the ML model is loaded and ready.")


class SignPredictionResponse(BaseModel):
    """Schema representing sign language classification output."""

    gloss: str = Field(..., description="Predicted sign language gloss label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0.0, 1.0].",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative candidate gloss labels, if any.",
    )
    below_threshold: bool = Field(
        ...,
        description="Flag indicating if prediction confidence is below the threshold.",
    )
