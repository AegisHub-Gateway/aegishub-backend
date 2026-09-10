"""
Scroll's AI Interface — Model Registry
=======================================
A single, modular entry point for every inference model used across
AegisHub. Each model is loaded lazily (on first use) and cached for the
lifetime of the process.

Right now no trained checkpoints exist, so every `predict_*` method
transparently falls back to a deterministic dummy predictor. Once
Scroll drops trained `.pt` weights into `app/models/weights/`, dropping
them at the paths configured in `app/core/config.py` is enough to
switch a given model over to real inference — no router or schema code
needs to change, since the public method signatures and return shapes
are identical in both modes.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger("aegishub.models")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard requirement, but degrade gracefully
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not installed — running in dummy-inference-only mode.")


# ---------------------------------------------------------------------------
# Static label tables (stand-ins until real class mappings ship with weights)
# ---------------------------------------------------------------------------

SIGN_LABELS = [
    "HELLO", "THANK_YOU", "PLEASE", "YES", "NO",
    "HELP", "EMERGENCY", "PAIN", "DOCTOR", "WATER",
]
EMERGENCY_SIGNS = {"HELP", "EMERGENCY", "PAIN"}

DERMA_CONDITIONS = [
    ("Benign Nevus", "low"),
    ("Seborrheic Keratosis", "low"),
    ("Eczema", "low"),
    ("Psoriasis", "medium"),
    ("Atypical Mole", "medium"),
    ("Suspected Melanoma", "high"),
]

LIPREAD_PHRASES = [
    "I need help please",
    "Where is the nearest exit",
    "Call a doctor immediately",
    "I am feeling dizzy",
    "Thank you for your assistance",
    "Can you repeat that please",
]


class ModelRegistry:
    """
    Lazily loads and caches every ML model used across AegisHub, and
    exposes a uniform `predict_*` API regardless of whether real weights
    or dummy fallbacks are backing a given model.
    """

    def __init__(self) -> None:
        self._sign_model: Optional[Any] = None
        self._derma_model: Optional[Any] = None
        self._sign_model_loaded = False
        self._derma_model_loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_sign_model(self) -> Optional[Any]:
        """Attempt to load Scroll's LSTM sign-language checkpoint."""
        if self._sign_model_loaded:
            return self._sign_model

        self._sign_model_loaded = True
        weights_path = Path(settings.SIGN_MODEL_PATH)

        if TORCH_AVAILABLE and weights_path.exists():
            try:
                self._sign_model = torch.load(weights_path, map_location="cpu")
                self._sign_model.eval()
                logger.info("Loaded LSTM sign model from %s", weights_path)
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception("Failed to load sign model weights from %s", weights_path)
                self._sign_model = None
        else:
            logger.info(
                "Sign model weights not found at %s — using dummy predictor.", weights_path
            )
            self._sign_model = None

        return self._sign_model

    def _load_derma_model(self) -> Optional[Any]:
        """Attempt to load Scroll's MobileNet derma-scan checkpoint."""
        if self._derma_model_loaded:
            return self._derma_model

        self._derma_model_loaded = True
        weights_path = Path(settings.DERMA_MODEL_PATH)

        if TORCH_AVAILABLE and weights_path.exists():
            try:
                self._derma_model = torch.load(weights_path, map_location="cpu")
                self._derma_model.eval()
                logger.info("Loaded MobileNet derma model from %s", weights_path)
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception("Failed to load derma model weights from %s", weights_path)
                self._derma_model = None
        else:
            logger.info(
                "Derma model weights not found at %s — using dummy predictor.", weights_path
            )
            self._derma_model = None

        return self._derma_model

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------
    def predict_sign(self, landmark_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Classify a `(frames, 21, 3)` hand-landmark tensor into a sign label.

        Returns:
            {"detected_sign": str, "confidence": float, "is_emergency": bool}
        """
        model = self._load_sign_model()

        if model is None:
            return self._dummy_predict_sign(landmark_sequence)

        # --- Real inference path (activates once a checkpoint is present) ---
        with torch.no_grad():
            tensor = torch.tensor(landmark_sequence, dtype=torch.float32).unsqueeze(0)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            confidence, idx = torch.max(probs, dim=0)
            label = SIGN_LABELS[int(idx)] if int(idx) < len(SIGN_LABELS) else "UNKNOWN"
            return {
                "detected_sign": label,
                "confidence": round(float(confidence), 4),
                "is_emergency": label in EMERGENCY_SIGNS,
            }

    def predict_derma(self, image_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Classify a preprocessed `(224, 224, 3)` normalized image.

        Returns:
            {"condition": str, "urgency": "low"|"medium"|"high", "confidence": float}
        """
        model = self._load_derma_model()

        if model is None:
            return self._dummy_predict_derma(image_tensor)

        # --- Real inference path (activates once a checkpoint is present) ---
        with torch.no_grad():
            tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            confidence, idx = torch.max(probs, dim=0)
            condition, urgency = DERMA_CONDITIONS[int(idx) % len(DERMA_CONDITIONS)]
            return {
                "condition": condition,
                "urgency": urgency,
                "confidence": round(float(confidence), 4),
            }

    def dummy_predict_lipread(self, transcript_seed: bytes, lip_mesh_frame_count: int) -> Dict[str, Any]:
        """
        Placeholder audio-visual lip-reading predictor. Real inference would
        fuse the raw audio waveform with the lip-mesh motion sequence via a
        multimodal model; this stands in until that model ships.

        Returns:
            {"transcript": str, "confidence": float, "is_muffled": bool}
        """
        seed = (sum(transcript_seed[:64]) if transcript_seed else 0) + lip_mesh_frame_count
        rng = random.Random(seed)
        transcript = rng.choice(LIPREAD_PHRASES)
        confidence = round(rng.uniform(0.60, 0.96), 4)
        return {
            "transcript": transcript,
            "confidence": confidence,
            "is_muffled": confidence < 0.75,
        }

    # ------------------------------------------------------------------
    # Dummy fallback predictors
    # ------------------------------------------------------------------
    @staticmethod
    def _dummy_predict_sign(landmark_sequence: np.ndarray) -> Dict[str, Any]:
        """Deterministic-per-input placeholder standing in for the LSTM model."""
        seed = int(abs(np.sum(landmark_sequence)) * 1000) % (2**32 - 1)
        rng = random.Random(seed)
        label = rng.choice(SIGN_LABELS)
        confidence = round(rng.uniform(0.72, 0.98), 4)
        return {
            "detected_sign": label,
            "confidence": confidence,
            "is_emergency": label in EMERGENCY_SIGNS,
        }

    @staticmethod
    def _dummy_predict_derma(image_tensor: np.ndarray) -> Dict[str, Any]:
        """Deterministic-per-input placeholder standing in for the MobileNet model."""
        seed = int(abs(np.sum(image_tensor)) * 1000) % (2**32 - 1)
        rng = random.Random(seed)
        condition, urgency = rng.choice(DERMA_CONDITIONS)
        confidence = round(rng.uniform(0.65, 0.95), 4)
        return {
            "condition": condition,
            "urgency": urgency,
            "confidence": confidence,
        }


_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Return the process-wide singleton `ModelRegistry` (FastAPI dependency-friendly)."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
