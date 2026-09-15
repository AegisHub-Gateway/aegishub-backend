"""Application configuration settings for AegisHub Sign Language API."""

import json
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    APP_NAME: str = Field(default="aegishub-sign-api", description="Service name")
    APP_ENV: str = Field(default="development", description="Runtime environment")
    MODEL_VERSION: str = Field(default="not-loaded", description="Loaded model version tag")
    MODEL_LOADED: bool = Field(default=False, description="Flag indicating if model is ready")

    # Phase 4 Model Adapter Settings
    MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Path to production weights checkpoint file (unloaded in mock mode)",
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which predictions are marked below_threshold",
    )
    CLASS_LABELS: list[str] = Field(
        default_factory=lambda: ["none"],
        description="Configurable classification labels list",
    )
    MOCK_MODEL_ENABLED: bool = Field(
        default=True,
        description="Whether mock model is enabled for local development",
    )

    @field_validator("CLASS_LABELS", mode="before")
    @classmethod
    def parse_class_labels(cls, value: Any) -> list[str]:
        """Support comma-separated strings or JSON arrays in environment variables."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


settings: Settings = get_settings()
