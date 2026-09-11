import os
import sys

# Ensure repository root is on sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app.config import Settings
from app.model_adapter import ModelNotLoadedError, SignModelAdapter
from app.schemas import Frame, Landmark, SignInput
from app.sign_service import SignService


def _create_mock_sign_input() -> SignInput:
    """Create sample landmark input for testing prediction pipeline."""
    wrist = Landmark(x=0.2, y=0.3, z=0.0)
    frame = Frame(left_hand=[wrist] * 21, right_hand=[])
    return SignInput(frames=[frame] * 30)


def test_mock_model_loads() -> None:
    """Verify mock model loads successfully and updates its status."""
    cfg = Settings(MOCK_MODEL_ENABLED=True, MODEL_LOADED=False)
    adapter = SignModelAdapter(config=cfg)

    assert adapter.is_loaded() is False
    load_success = adapter.load()
    assert load_success is True
    assert adapter.is_loaded() is True


def test_model_status_reported_correctly() -> None:
    """Verify is_loaded accurately reflects initialization state."""
    cfg_unloaded = Settings(MOCK_MODEL_ENABLED=True, MODEL_LOADED=False)
    adapter_unloaded = SignModelAdapter(config=cfg_unloaded)
    assert adapter_unloaded.is_loaded() is False

    cfg_loaded = Settings(MOCK_MODEL_ENABLED=True, MODEL_LOADED=True)
    adapter_loaded = SignModelAdapter(config=cfg_loaded)
    assert adapter_loaded.is_loaded() is True


def test_prediction_returns_expected_structure() -> None:
    """Verify prediction produces expected structure matching the mock specification."""
    cfg = Settings(MOCK_MODEL_ENABLED=True, MODEL_LOADED=True)
    adapter = SignModelAdapter(config=cfg)

    result = adapter.predict(input_data=None)

    assert isinstance(result, dict)
    assert "gloss" in result
    assert "confidence" in result
    assert "alternatives" in result
    assert "below_threshold" in result

    assert result["gloss"] == "none"
    assert result["confidence"] == 0.0
    assert result["alternatives"] == []
    assert result["below_threshold"] is True


def test_missing_model_handled_safely() -> None:
    """Verify non-existent real model path does not crash the application."""
    cfg_missing = Settings(
        MOCK_MODEL_ENABLED=False,
        MODEL_PATH="models/non_existent_weights.pt",
        MODEL_LOADED=False,
    )
    adapter = SignModelAdapter(config=cfg_missing)

    # Calling load on missing file returns False without unhandled exception
    loaded = adapter.load()
    assert loaded is False
    assert adapter.is_loaded() is False

    # Calling predict on unloaded model raises ModelNotLoadedError
    with pytest.raises(ModelNotLoadedError, match="model is not currently loaded"):
        adapter.predict(input_data=None)


def test_model_version_returned() -> None:
    """Verify model adapter returns the configured version string."""
    cfg = Settings(MODEL_VERSION="scroll-lstm-v0.1-dev", MOCK_MODEL_ENABLED=True)
    adapter = SignModelAdapter(config=cfg)

    assert adapter.get_model_version() == "scroll-lstm-v0.1-dev"


def test_low_confidence_handled_correctly() -> None:
    """Verify confidence threshold correctly flags below_threshold predictions."""
    cfg = Settings(
        MOCK_MODEL_ENABLED=True,
        MODEL_LOADED=True,
        CONFIDENCE_THRESHOLD=0.70,
    )
    adapter = SignModelAdapter(config=cfg)

    # Case 1: Confidence 0.45 is below threshold 0.70 -> below_threshold: True
    low_result = adapter.predict(
        input_data=None,
        simulated_gloss="help",
        simulated_confidence=0.45,
    )
    assert low_result["gloss"] == "help"
    assert low_result["confidence"] == 0.45
    assert low_result["below_threshold"] is True

    # Case 2: Confidence 0.92 is above threshold 0.70 -> below_threshold: False
    high_result = adapter.predict(
        input_data=None,
        simulated_gloss="pain",
        simulated_confidence=0.92,
    )
    assert high_result["gloss"] == "pain"
    assert high_result["confidence"] == 0.92
    assert high_result["below_threshold"] is False


def test_sign_service_predict_integration() -> None:
    """Verify SignService orchestrates preprocessing and adapter prediction."""
    cfg = Settings(MOCK_MODEL_ENABLED=True, MODEL_LOADED=True)
    adapter = SignModelAdapter(config=cfg)
    service = SignService(adapter=adapter)

    sample_input = _create_mock_sign_input()
    prediction = service.predict_sign(sample_input)

    assert prediction["gloss"] == "none"
    assert prediction["confidence"] == 0.0
    assert prediction["below_threshold"] is True
