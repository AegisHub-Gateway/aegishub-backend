import os
import sys

# Ensure repository root is on sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from app.preprocessing import (
    DEFAULT_FRAMES,
    MissingHandPolicy,
    convert_to_model_input,
    normalize_hand_landmarks,
    normalize_sequence,
    sign_input_to_numpy,
    validate_sequence_shape,
)
from app.schemas import Frame, Landmark, SignInput


def _create_synthetic_hand(wrist_x: float, wrist_y: float, wrist_z: float, spread: float = 0.1) -> list[Landmark]:
    """Generate 21 synthetic landmarks starting at wrist (wrist_x, wrist_y, wrist_z)."""
    landmarks = [Landmark(x=wrist_x, y=wrist_y, z=wrist_z)]
    for i in range(1, 21):
        offset = spread * (i / 21.0)
        landmarks.append(
            Landmark(
                x=min(1.0, max(0.0, wrist_x + offset)),
                y=min(1.0, max(0.0, wrist_y + offset)),
                z=wrist_z + offset,
            )
        )
    return landmarks


def _create_30_frame_sign_input() -> SignInput:
    """Generate a valid 30-frame SignInput payload."""
    frames = []
    for f in range(DEFAULT_FRAMES):
        left = _create_synthetic_hand(0.2, 0.3, -0.02, spread=0.15)
        right = _create_synthetic_hand(0.6, 0.4, 0.01, spread=0.12)
        frames.append(Frame(left_hand=left, right_hand=right))
    return SignInput(frames=frames)


def test_correct_output_shapes() -> None:
    """Verify convert_to_model_input produces (30, 2, 21, 3) and flattened (30, 126)."""
    sign_input = _create_30_frame_sign_input()

    # Unflattened shape: [30, 2, 21, 3]
    tensor_4d = convert_to_model_input(sign_input, flatten=False)
    assert isinstance(tensor_4d, np.ndarray)
    assert tensor_4d.shape == (30, 2, 21, 3)
    assert tensor_4d.dtype == np.float32

    # Flattened shape: [30, 126] (30 frames * (2 * 21 * 3))
    tensor_2d = convert_to_model_input(sign_input, flatten=True)
    assert isinstance(tensor_2d, np.ndarray)
    assert tensor_2d.shape == (30, 126)
    assert tensor_2d.dtype == np.float32


def test_wrist_becomes_approximately_zero() -> None:
    """Verify that wrist normalization shifts landmark 0 to approximately [0, 0, 0]."""
    hand_coords = np.array(
        [[0.35, 0.45, -0.12]] + [[0.35 + i * 0.02, 0.45 + i * 0.01, -0.12 + i * 0.01] for i in range(1, 21)],
        dtype=np.float32,
    )

    normalized = normalize_hand_landmarks(hand_coords)

    # Wrist (landmark 0) must be within floating-point tolerance of [0, 0, 0]
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_scale_normalization_works() -> None:
    """Verify scale normalization produces maximum distance of 1.0 and is scale-invariant."""
    # Hand with wrist at (0.2, 0.2, 0.0) and tip at (0.6, 0.5, 0.0) -> distance = sqrt(0.4^2 + 0.3^2) = 0.5
    wrist = [0.2, 0.2, 0.0]
    tip = [0.6, 0.5, 0.0]
    intermediate = [[0.2 + i * 0.01, 0.2 + i * 0.01, 0.0] for i in range(1, 20)]
    hand = np.array([wrist] + intermediate + [tip], dtype=np.float32)

    normalized = normalize_hand_landmarks(hand)

    # Maximum distance from wrist across all landmarks must be exactly 1.0
    distances = np.linalg.norm(normalized - normalized[0], axis=-1)
    assert pytest.approx(float(np.max(distances)), rel=1e-5) == 1.0

    # Scale invariance: multiplying coordinates around wrist by factor k yields identical normalized output
    scaled_hand = hand.copy()
    scaled_hand[1:] = wrist + (hand[1:] - wrist) * 3.5
    normalized_scaled = normalize_hand_landmarks(scaled_hand)
    np.testing.assert_allclose(normalized, normalized_scaled, atol=1e-5)


def test_zero_size_hand_does_not_crash() -> None:
    """Verify that a zero-scale hand (identical points) does not raise ZeroDivisionError."""
    # All 21 landmarks at exact same coordinate
    identical_points = np.full((21, 3), fill_value=0.5, dtype=np.float32)

    # Must safely fallback and not crash
    result = normalize_hand_landmarks(identical_points)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.zeros((21, 3), dtype=np.float32), atol=1e-6)


def test_sequence_order_is_preserved() -> None:
    """Verify that the temporal frame order (0 to 29) is strictly preserved."""
    frames = []
    for f in range(DEFAULT_FRAMES):
        # Unique distinguishable marker in wrist coordinate for each frame
        left = _create_synthetic_hand(f / 100.0, 0.3, 0.0, spread=0.05)
        right = _create_synthetic_hand(0.5, f / 100.0, 0.0, spread=0.05)
        frames.append(Frame(left_hand=left, right_hand=right))

    sign_input = SignInput(frames=frames)
    raw_array = sign_input_to_numpy(sign_input)

    # Verify raw extracted array preserved frame order
    for f in range(DEFAULT_FRAMES):
        assert pytest.approx(raw_array[f, 0, 0, 0], abs=1e-5) == f / 100.0
        assert pytest.approx(raw_array[f, 1, 0, 1], abs=1e-5) == f / 100.0

    # Verify normalize_sequence preserves frame indexing
    normalized = normalize_sequence(raw_array)
    assert normalized.shape[0] == DEFAULT_FRAMES


def test_invalid_input_is_rejected() -> None:
    """Verify validate_sequence_shape and convert_to_model_input reject malformed inputs."""
    # Wrong type
    with pytest.raises(TypeError):
        validate_sequence_shape("not an array")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        convert_to_model_input({"frames": []})  # type: ignore[arg-type]

    # Wrong frame count (e.g. 15 instead of 30)
    bad_shape_array = np.zeros((15, 2, 21, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid sequence shape"):
        validate_sequence_shape(bad_shape_array, expected_shape=(30, 2, 21, 3))

    with pytest.raises(ValueError, match="Invalid sequence shape"):
        convert_to_model_input(bad_shape_array)


def test_missing_hand_behavior_is_consistent() -> None:
    """Verify missing hand policy produces consistent zero-filled landmarks."""
    # Frame with only left hand present, right hand empty
    left = _create_synthetic_hand(0.3, 0.3, 0.0)
    frame = Frame(left_hand=left, right_hand=[])
    sign_input = SignInput(frames=[frame] * DEFAULT_FRAMES)

    array = convert_to_model_input(sign_input, missing_hand_policy=MissingHandPolicy.ZEROS)

    # Left hand is normalized with wrist at 0
    np.testing.assert_allclose(array[:, 0, 0], np.zeros((DEFAULT_FRAMES, 3)), atol=1e-6)

    # Right hand (missing) is cleanly represented as zeros across all 21 landmarks
    np.testing.assert_allclose(array[:, 1], np.zeros((DEFAULT_FRAMES, 21, 3)), atol=1e-6)
