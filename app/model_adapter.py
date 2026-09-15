"""Model adapter layer for the AegisHub Sign Language API.

Decouples the FastAPI service from the internal implementation details of
future AI models (e.g. Scroll's LSTM/GRU) and provides a mock model for
development testing.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.config import Settings, get_settings
from app.schemas import SignPredictionResponse

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is requested on an unloaded model adapter."""


class BaseModelInterface(ABC):
    """Abstract interface for sign classification models."""

    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load model weights or resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded and ready."""

    @abstractmethod
    def predict(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform inference on input data."""

    @abstractmethod
    def get_version(self) -> str:
        """Return model version string."""


class MockSignModel(BaseModelInterface):
    """Clearly marked mock model for development and testing only.

    NOTE: Does NOT represent real AI model predictions or accuracy.
    Returns deterministic or simulated outputs for end-to-end integration.
    """

    def __init__(
        self,
        version: str = "mock-v1",
        confidence_threshold: float = 0.6,
        class_labels: Optional[list[str]] = None,
    ) -> None:
        self.version = version
        self.confidence_threshold = confidence_threshold
        self.class_labels = class_labels or ["none"]
        self._loaded: bool = False

    def load(self, model_path: Optional[str] = None) -> bool:
        """Simulate model loading for local development."""
        self._loaded = True
        logger.info("MockSignModel loaded successfully in development mock mode.")
        return True

    def is_loaded(self) -> bool:
        """Return whether mock model is initialized."""
        return self._loaded

    def get_version(self) -> str:
        """Return mock model version tag."""
        return self.version

    def predict(
        self,
        input_data: Any = None,
        simulated_gloss: str = "none",
        simulated_confidence: float = 0.0,
        simulated_alternatives: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Return mock inference result.

        Evaluates confidence against the configured confidence_threshold to
        accurately flag low confidence.
        """
        if not self._loaded:
            raise ModelNotLoadedError("MockSignModel is not loaded. Call load() first.")

        below_threshold = simulated_confidence < self.confidence_threshold
        alternatives = simulated_alternatives if simulated_alternatives is not None else []

        return {
            "gloss": simulated_gloss,
            "confidence": float(simulated_confidence),
            "alternatives": alternatives,
            "below_threshold": below_threshold,
        }


class ScrollModelPlaceholder(BaseModelInterface):
    """Placeholder integration for Scroll's future trained LSTM/GRU model.

    Leaves input shapes, class order, filename, and preprocessing uninvented
    until confirmed by the model engineering team. Safely handles missing weights.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        version: str = "unloaded",
        confidence_threshold: float = 0.6,
        class_labels: Optional[list[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.version = version
        self.confidence_threshold = confidence_threshold
        self.class_labels = class_labels or []
        self._loaded: bool = False
        self._model_instance: Any = None

    def load(self, model_path: Optional[str] = None) -> bool:
        """Safely attempt to locate and load the model weights checkpoint."""
        target_path = model_path or self.model_path
        if not target_path or not os.path.exists(target_path):
            logger.warning(
                "Scroll model weights not found at path: '%s'. Model remains unloaded.",
                target_path,
            )
            self._loaded = False
            return False

        # Placeholder: When weights are provided by Scroll, model instantiation occurs here.
        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        """Return whether the real model weights are loaded."""
        return self._loaded

    def get_version(self) -> str:
        """Return model version."""
        return self.version

    def predict(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Inference placeholder raising an error if weights are uninstantiated."""
        if not self._loaded:
            raise ModelNotLoadedError(
                "Scroll's model is not loaded. Ensure valid model weights are provided."
            )
        raise NotImplementedError(
            "Real model inference is not yet implemented pending Scroll's model checkpoint."
        )


class SignModelAdapter:
    """Unified adapter decoupling API services from underlying model implementations."""

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.config: Settings = config or get_settings()
        self._backend: BaseModelInterface

        if self.config.MOCK_MODEL_ENABLED:
            self._backend = MockSignModel(
                version=self.config.MODEL_VERSION,
                confidence_threshold=self.config.CONFIDENCE_THRESHOLD,
                class_labels=self.config.CLASS_LABELS,
            )
        else:
            self._backend = ScrollModelPlaceholder(
                model_path=self.config.MODEL_PATH,
                version=self.config.MODEL_VERSION,
                confidence_threshold=self.config.CONFIDENCE_THRESHOLD,
                class_labels=self.config.CLASS_LABELS,
            )

        # If settings specify MODEL_LOADED=True, trigger load immediately
        if self.config.MODEL_LOADED:
            self.load()

    def load(self, model_path: Optional[str] = None) -> bool:
        """Load the configured model backend."""
        path = model_path or self.config.MODEL_PATH
        return self._backend.load(path)

    def is_loaded(self) -> bool:
        """Report whether the underlying model is loaded."""
        return self._backend.is_loaded()

    def get_model_version(self) -> str:
        """Return the active model version."""
        return self._backend.get_version()

    def predict(self, input_data: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Run prediction through the adapter with threshold validation.

        Returns a dictionary conforming to SignPredictionResponse.
        """
        if not self.is_loaded():
            raise ModelNotLoadedError(
                "Cannot perform prediction: model is not currently loaded."
            )
        raw_result = self._backend.predict(input_data, **kwargs)

        # Validate through Pydantic schema for contract integrity
        validated = SignPredictionResponse.model_validate(raw_result)
        return validated.model_dump()


_default_adapter: Optional[SignModelAdapter] = None


def get_model_adapter() -> SignModelAdapter:
    """Return singleton instance of SignModelAdapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = SignModelAdapter()
    return _default_adapter
