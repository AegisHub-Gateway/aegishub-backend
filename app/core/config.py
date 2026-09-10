"""
Core Settings
=============
Centralized, environment-driven configuration for AegisHub Gateway using
Pydantic v2's `pydantic-settings` package. All values can be overridden
via environment variables or a `.env` file at the project root without
touching code.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration singleton."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Project Metadata ---
    PROJECT_NAME: str = "AegisHub Gateway"
    PROJECT_DESCRIPTION: str = (
        "Multi-modal health accessibility gateway providing AI-powered "
        "sign-language recognition, derma-scan triage, and lip-reading "
        "inference. Built for GatewayHacks 2026."
    )
    VERSION: str = "1.0.0"

    # --- API Configuration ---
    API_V1_PREFIX: str = "/api/v1"

    # --- Server Configuration (used by `uvicorn` when run via `python -m app.main`) ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # --- CORS ---
    # Wide open for hackathon/local development so the Next.js frontend
    # can call the API from any origin/port. Lock this down before any
    # production deployment.
    CORS_ORIGINS: List[str] = ["*"]

    # --- Model Weights (Scroll's AI interface) ---
    # Paths are resolved relative to the process working directory. Files
    # need not exist yet — `ModelRegistry` gracefully falls back to dummy
    # predictors when a checkpoint is missing.
    MODEL_WEIGHTS_DIR: str = "app/models/weights"
    SIGN_MODEL_PATH: str = "app/models/weights/sign_lstm.pt"
    DERMA_MODEL_PATH: str = "app/models/weights/derma_mobilenet.pt"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached `Settings` instance."""
    return Settings()


settings = get_settings()
