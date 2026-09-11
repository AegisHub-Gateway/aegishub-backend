"""Landmark preprocessing pipeline for the AegisHub Sign Language API.

Converts validated sign-language landmark requests into model-ready numerical
representations (NumPy arrays) with wrist-centering and scale normalization.
"""

from enum import Enum

import numpy as np

from app.schemas import Frame, Landmark, SignInput

# Default model input dimensions
DEFAULT_FRAMES: int = 30
DEFAULT_HANDS: int = 2
DEFAULT_LANDMARKS_PER_HAND: int = 21
DEFAULT_COORDS_PER_LANDMARK: int = 3
DEFAULT_EPSILON: float = 1e-6


class MissingHandPolicy(str, Enum):
    """Configurable policy for handling absent hands in landmark frames."""

    ZEROS = "zeros"


def validate_sign_input(sign_input: SignInput) -> SignInput:
    """Validate that input is a valid SignInput instance without mutating data."""
    if not isinstance(sign_input, SignInput):
        raise TypeError(f"Expected SignInput instance, got {type(sign_input).__name__}")
    return sign_input


def get_landmark_counts(sign_input: SignInput) -> list[dict[str, int]]:
    """Return counts of left and right hand landmarks per frame without altering values."""
    return [
        {
            "left_hand": len(frame.left_hand),
            "right_hand": len(frame.right_hand),
        }
        for frame in sign_input.frames
    ]


def _hand_to_numpy(
    landmarks: list[Landmark],
    target_landmarks: int = DEFAULT_LANDMARKS_PER_HAND,
    missing_hand_policy: MissingHandPolicy = MissingHandPolicy.ZEROS,
) -> np.ndarray:
    """Convert a list of Landmark objects for a single hand into a (21, 3) float32 array."""
    if not landmarks:
        if missing_hand_policy == MissingHandPolicy.ZEROS:
            return np.zeros((target_landmarks, DEFAULT_COORDS_PER_LANDMARK), dtype=np.float32)
        raise ValueError(f"Unsupported missing hand policy: {missing_hand_policy}")

    raw_coords = [[lm.x, lm.y, lm.z] for lm in landmarks]

    if len(raw_coords) == target_landmarks:
        return np.array(raw_coords, dtype=np.float32)
    elif len(raw_coords) < target_landmarks:
        padded = np.zeros((target_landmarks, DEFAULT_COORDS_PER_LANDMARK), dtype=np.float32)
        padded[: len(raw_coords)] = raw_coords
        return padded
    else:
        return np.array(raw_coords[:target_landmarks], dtype=np.float32)


def sign_input_to_numpy(
    sign_input: SignInput,
    expected_frames: int = DEFAULT_FRAMES,
    landmarks_per_hand: int = DEFAULT_LANDMARKS_PER_HAND,
    missing_hand_policy: MissingHandPolicy = MissingHandPolicy.ZEROS,
    pad_or_truncate: bool = False,
) -> np.ndarray:
    """Convert a SignInput instance into a NumPy array of shape (N, 2, 21, 3).

    Preserves the temporal sequence of frames. Index 0 corresponds to left_hand,
    and index 1 corresponds to right_hand.

    Args:
        sign_input: Validated SignInput container.
        expected_frames: Target number of frames.
        landmarks_per_hand: Number of landmarks expected per hand.
        missing_hand_policy: Policy for handling missing hands.
        pad_or_truncate: If True, pads or truncates the frame sequence to expected_frames.

    Returns:
        NumPy float32 array with shape (N, 2, 21, 3) or (expected_frames, 2, 21, 3).
    """
    validate_sign_input(sign_input)

    frames: list[Frame] = sign_input.frames
    frame_arrays: list[np.ndarray] = []

    for frame in frames:
        left = _hand_to_numpy(frame.left_hand, landmarks_per_hand, missing_hand_policy)
        right = _hand_to_numpy(frame.right_hand, landmarks_per_hand, missing_hand_policy)
        frame_arrays.append(np.stack([left, right], axis=0))

    if not frame_arrays:
        array = np.zeros((0, DEFAULT_HANDS, landmarks_per_hand, DEFAULT_COORDS_PER_LANDMARK), dtype=np.float32)
    else:
        array = np.stack(frame_arrays, axis=0).astype(np.float32)

    if pad_or_truncate:
        current_len = array.shape[0]
        if current_len < expected_frames:
            padding = np.zeros(
                (expected_frames - current_len, DEFAULT_HANDS, landmarks_per_hand, DEFAULT_COORDS_PER_LANDMARK),
                dtype=np.float32,
            )
            array = np.concatenate([array, padding], axis=0) if current_len > 0 else padding
        elif current_len > expected_frames:
            array = array[:expected_frames]

    return array


def validate_sequence_shape(
    sequence: np.ndarray,
    expected_shape: tuple[int, ...] = (
        DEFAULT_FRAMES,
        DEFAULT_HANDS,
        DEFAULT_LANDMARKS_PER_HAND,
        DEFAULT_COORDS_PER_LANDMARK,
    ),
) -> np.ndarray:
    """Confirm that the sequence array matches the expected multidimensional shape.

    Args:
        sequence: NumPy array to validate.
        expected_shape: Expected shape tuple, default (30, 2, 21, 3).

    Returns:
        The validated sequence array unchanged.

    Raises:
        TypeError: If sequence is not a np.ndarray.
        ValueError: If sequence.shape does not equal expected_shape.
    """
    if not isinstance(sequence, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(sequence).__name__}")

    if sequence.shape != expected_shape:
        raise ValueError(
            f"Invalid sequence shape {sequence.shape}; expected {expected_shape}."
        )

    return sequence


def normalize_hand_landmarks(
    hand_landmarks: np.ndarray,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Normalize a single hand's landmarks around the wrist with distance scaling.

    Mathematical behavior:
        1. Wrist normalization:
           wrist_centered = landmark - wrist
           (where wrist is index 0: landmarks[0])
        2. Scale normalization:
           normalized_landmark = wrist_centered / hand_scale
           (where hand_scale is the max distance from wrist across all landmarks)

    Fallback safety:
        If hand_scale is zero or below eps, returns wrist_centered without
        dividing by zero.

    Args:
        hand_landmarks: Array of shape (21, 3) where index 0 is the wrist.
        eps: Epsilon guard threshold against division by zero.

    Returns:
        Normalized array of shape (21, 3) with wrist at approximately [0, 0, 0].
    """
    if not isinstance(hand_landmarks, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(hand_landmarks).__name__}")

    if hand_landmarks.ndim != 2 or hand_landmarks.shape[-1] != DEFAULT_COORDS_PER_LANDMARK:
        raise ValueError(
            f"Expected hand landmarks with shape (L, 3), got {hand_landmarks.shape}"
        )

    if len(hand_landmarks) == 0:
        return hand_landmarks.copy()

    # Step 1: Wrist centering (landmark 0 is the wrist)
    wrist = hand_landmarks[0]
    wrist_centered = hand_landmarks - wrist

    # Step 2: Hand scale computation (max Euclidean distance from wrist)
    distances = np.linalg.norm(wrist_centered, axis=-1)
    hand_scale = float(np.max(distances))

    # Safe fallback when hand scale is zero or too small to prevent division by zero
    if hand_scale < eps:
        return wrist_centered.astype(np.float32)

    normalized = wrist_centered / hand_scale
    return normalized.astype(np.float32)


def normalize_sequence(
    sequence: np.ndarray,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Apply wrist-centering and scale normalization across all hands in all frames.

    Preserves temporal frame sequence order.

    Args:
        sequence: Array of shape (F, 2, 21, 3).
        eps: Epsilon guard against division by zero.

    Returns:
        Normalized array of shape (F, 2, 21, 3).
    """
    if not isinstance(sequence, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(sequence).__name__}")

    if sequence.ndim != 4 or sequence.shape[1:] != (
        DEFAULT_HANDS,
        DEFAULT_LANDMARKS_PER_HAND,
        DEFAULT_COORDS_PER_LANDMARK,
    ):
        raise ValueError(
            f"Expected sequence of shape (F, 2, 21, 3), got {sequence.shape}"
        )

    num_frames = sequence.shape[0]
    normalized = np.zeros_like(sequence, dtype=np.float32)

    for f in range(num_frames):
        for h in range(DEFAULT_HANDS):
            normalized[f, h] = normalize_hand_landmarks(sequence[f, h], eps=eps)

    return normalized


def convert_to_model_input(
    data: SignInput | np.ndarray,
    target_frames: int = DEFAULT_FRAMES,
    flatten: bool = False,
    missing_hand_policy: MissingHandPolicy = MissingHandPolicy.ZEROS,
    pad_or_truncate: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Convert validated landmark data into a model-ready numerical representation.

    Full pipeline:
    1. Ingests SignInput or raw NumPy array, preserving frame order.
    2. Converts to NumPy array with shape (30, 2, 21, 3).
    3. Applies missing hand policy (zeros for missing hands).
    4. Validates multidimensional shape.
    5. Applies wrist-centering and scale normalization.
    6. Configurably outputs shape [30, 2, 21, 3] or flattened [30, 126].

    Args:
        data: Validated SignInput model or pre-formed NumPy sequence.
        target_frames: Expected frame count (default: 30).
        flatten: If True, flattens (30, 2, 21, 3) to (30, 126).
        missing_hand_policy: Policy for missing hands (default: ZEROS).
        pad_or_truncate: Whether to pad/truncate frames to target_frames.
        eps: Epsilon threshold for safe division in scale normalization.

    Returns:
        Normalized NumPy float32 array of shape (30, 2, 21, 3) or (30, 126).
    """
    if isinstance(data, SignInput):
        array = sign_input_to_numpy(
            data,
            expected_frames=target_frames,
            landmarks_per_hand=DEFAULT_LANDMARKS_PER_HAND,
            missing_hand_policy=missing_hand_policy,
            pad_or_truncate=pad_or_truncate,
        )
    elif isinstance(data, np.ndarray):
        array = data.astype(np.float32)
    else:
        raise TypeError(f"Expected SignInput or np.ndarray, got {type(data).__name__}")

    # Confirm the shape before normalization
    expected_shape = (
        target_frames,
        DEFAULT_HANDS,
        DEFAULT_LANDMARKS_PER_HAND,
        DEFAULT_COORDS_PER_LANDMARK,
    )
    validate_sequence_shape(array, expected_shape)

    # Normalize sequence (wrist centering + scale normalization per hand)
    normalized = normalize_sequence(array, eps=eps)

    # Configurable output shape: [30, 2, 21, 3] or [30, 126]
    if flatten:
        return normalized.reshape(target_frames, -1)

    return normalized
