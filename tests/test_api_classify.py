import os
import sys

# Ensure repository root is on sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_classify_endpoint_success() -> None:
    """Verify POST /v1/sign/classify receives landmarks and returns expected prediction."""
    sample_payload = {
        "frames": [
            {
                "left_hand": [
                    {"x": 0.12, "y": 0.35, "z": -0.02} for _ in range(21)
                ],
                "right_hand": [
                    {"x": 0.18, "y": 0.31, "z": -0.01} for _ in range(21)
                ],
            }
            for _ in range(30)
        ]
    }

    response = client.post("/v1/sign/classify", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["gloss"] == "none"
    assert data["confidence"] == 0.0
    assert data["alternatives"] == []
    assert data["below_threshold"] is True


def test_classify_endpoint_invalid_landmarks_rejected() -> None:
    """Verify POST /v1/sign/classify rejects out-of-bounds coordinates with HTTP 422."""
    invalid_payload = {
        "frames": [
            {
                "left_hand": [{"x": 1.5, "y": 0.5, "z": 0.0}],  # x > 1.0
                "right_hand": [],
            }
        ]
    }

    response = client.post("/v1/sign/classify", json=invalid_payload)
    assert response.status_code == 422
