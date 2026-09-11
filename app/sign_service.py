"""Sign language service layer for the AegisHub Sign Language API.

Orchestrates preprocessing and model prediction workflows while keeping
the API decoupled from model internal details.
"""

from typing import Any

from app.model_adapter import SignModelAdapter, get_model_adapter
from app.preprocessing import convert_to_model_input, validate_sign_input
from app.schemas import SignInput


class SignService:
    """Service managing sign language preprocessing and prediction pipelines."""

    def __init__(self, adapter: SignModelAdapter | None = None) -> None:
        self.adapter = adapter or get_model_adapter()

    def handle_sign_input(self, sign_input: SignInput) -> SignInput:
        """Validate input landmarks without triggering inference."""
        return validate_sign_input(sign_input)

    def predict_sign(self, sign_input: SignInput, **kwargs: Any) -> dict[str, Any]:
        """Convert landmarks to model-ready tensor and perform inference through adapter."""
        if not self.adapter.is_loaded() and self.adapter.config.MOCK_MODEL_ENABLED:
            self.adapter.load()

        model_input = convert_to_model_input(
            sign_input,
            flatten=False,
            pad_or_truncate=True,
        )
        return self.adapter.predict(model_input, **kwargs)


_default_service: SignService | None = None


def get_sign_service() -> SignService:
    """Return singleton instance of SignService."""
    global _default_service
    if _default_service is None:
        _default_service = SignService()
    return _default_service
