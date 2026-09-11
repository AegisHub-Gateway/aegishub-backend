import cv2
import librosa
import numpy as np


def process_video_frames(video_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extracts frames from a video file, converts BGR to RGB,
    resizes to target dimensions, and normalizes pixels to [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR default to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to target dimensions
        resized_frame = cv2.resize(rgb_frame, target_size)

        # Normalize pixel values
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        frames.append(normalized_frame)

    cap.release()

    if not frames:
        raise ValueError("Could not read frames from video.")

    return np.array(frames, dtype=np.float32)


def process_lip_landmarks(lip_mesh_dict: dict) -> np.ndarray:
    """
    Extracts x, y coordinates from lip_mesh dictionary into a NumPy array.
    """
    frames = lip_mesh_dict.get("frames", [])
    mesh_sequence = []

    for frame in frames:
        points = frame.get("points", [])
        coords = [[pt["x"], pt["y"]] for pt in points]
        mesh_sequence.append(coords)

    return np.array(mesh_sequence, dtype=np.float32)


def process_audio_file(file_path: str, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Loads an audio file (.ogg/wav), resamples to target sample rate,
    and extracts MFCC (Mel-frequency cepstral coefficients) features.

    Args:
        file_path: Path to audio file (.ogg, .wav, etc.)
        sr: Target sample rate (default: 16000 Hz)
        n_mfcc: Number of MFCC coefficients to extract (default: 13)

    Returns:
        np.ndarray: MFCC feature matrix of shape (n_mfcc, time_steps)

    Raises:
        ValueError: If audio file cannot be processed
    """
    try:
        # Load audio and automatically convert to mono
        y, native_sr = librosa.load(file_path, sr=sr)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        return mfccs
    except Exception as e:
        raise ValueError(f"Failed to process audio file: {str(e)}")


def process_derma_image(image_bytes: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Decodes image bytes, converts BGR to RGB, resizes to target dimensions,
    and normalizes pixel values to [0.0, 1.0].

    Args:
        image_bytes: Raw image bytes from upload
        target_size: Target dimensions (width, height) for resizing

    Returns:
        np.ndarray: Normalized RGB image of shape (height, width, 3) with dtype float32

    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Decode raw bytes into NumPy BGR image via OpenCV
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if bgr_image is None:
            raise ValueError("Image buffer could not be decoded as a valid image")

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions
        resized_image = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to [0.0, 1.0]
        normalized_image = resized_image.astype(np.float32) / 255.0

        return normalized_image

    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")
