"""
Neche's Responsibility — Derma-Scan Schemas
============================================
Pydantic v2 models for the derma-scan image-triage endpoint. The image
itself streams in as multipart `UploadFile` (validated in the router);
these schemas cover the structured response contract returned after
classification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UrgencyLevel = Literal["low", "medium", "high"]


class DermaScanResponse(BaseModel):
    """Structured triage result returned after MobileNet classification."""

    condition: str = Field(..., description="Predicted dermatological condition label.")
    urgency: UrgencyLevel = Field(..., description="Triage urgency bucket.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    summary: str = Field(..., description="Human-readable, patient-facing summary of the result.")
