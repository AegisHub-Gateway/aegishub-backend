import os
import sys

# Ensure repository root is on sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.model_adapter import SignModelAdapter
from app.preprocessing import get_landmark_counts, validate_sign_input
from app.schemas import Frame, Landmark, SignInput
from app.sign_service import SignService

client = TestClient(app)


def test_healthz_endpoint() -> None:
    """Verify /healthz returns HTTP 200 and expected payload structure."""
    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "aegishub-sign-api"
    assert data["model_version"] == "not-loaded"
    assert data["model_loaded"] is False


def test_sample_landmark_structure_accepted() -> None:
    """Verify the sample landmark JSON structure is accepted by SignInput."""
    sample_payload = {
        "frames": [
            {
                "left_hand": [
                    {
                        "x": 0.12,
                        "y": 0.35,
                        "z": -0.02,
                    }
                ],
                "right_hand": [
                    {
                        "x": 0.18,
                        "y": 0.31,
                        "z": -0.01,
                    }
                ],
            }
        ]
    }

    sign_input = SignInput.model_validate(sample_payload)
    assert len(sign_input.frames) == 1

    frame = sign_input.frames[0]
    assert len(frame.left_hand) == 1
    assert len(frame.right_hand) == 1

    assert frame.left_hand[0].x == 0.12
    assert frame.left_hand[0].y == 0.35
    assert frame.left_hand[0].z == -0.02

    assert frame.right_hand[0].x == 0.18
    assert frame.right_hand[0].y == 0.31
    assert frame.right_hand[0].z == -0.01


def test_frame_with_empty_hand_list_accepted() -> None:
    """Verify frames with missing or empty hands are accepted."""
    # Both hands explicitly empty
    payload_both_empty = {
        "frames": [
            {
                "left_hand": [],
                "right_hand": [],
            }
        ]
    }
    parsed = SignInput.model_validate(payload_both_empty)
    assert len(parsed.frames) == 1
    assert parsed.frames[0].left_hand == []
    assert parsed.frames[0].right_hand == []

    # One hand detected, other hand empty
    payload_one_hand = {
        "frames": [
            {
                "left_hand": [{"x": 0.25, "y": 0.45, "z": 0.05}],
                "right_hand": [],
            }
        ]
    }
    parsed_one_hand = SignInput.model_validate(payload_one_hand)
    assert len(parsed_one_hand.frames[0].left_hand) == 1
    assert len(parsed_one_hand.frames[0].right_hand) == 0

    # Omitted hand keys default to empty list
    frame_default = Frame()
    assert frame_default.left_hand == []
    assert frame_default.right_hand == []


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [
        (-0.01, 0.5, 0.0),  # x below 0.0
        (1.01, 0.5, 0.0),   # x above 1.0
        (0.5, -0.05, 0.0),  # y below 0.0
        (0.5, 1.20, 0.0),   # y above 1.0
    ],
)
def test_invalid_normalized_coordinates_rejected(x: float, y: float, z: float) -> None:
    """Verify that coordinates outside normalized [0.0, 1.0] range are rejected."""
    with pytest.raises(ValidationError):
        Landmark(x=x, y=y, z=z)


def test_preprocessing_and_service_placeholders() -> None:
    """Verify preprocessing pass-through utility and service layer behave as expected."""
    sample = SignInput(
        frames=[
            Frame(
                left_hand=[Landmark(x=0.1, y=0.2, z=-0.5)],
                right_hand=[],
            )
        ]
    )

    # Preprocessing validation guard returns input unchanged
    assert validate_sign_input(sample) == sample
    assert get_landmark_counts(sample) == [{"left_hand": 1, "right_hand": 0}]

    # SignService handles input without predicting
    service = SignService()
    result = service.handle_sign_input(sample)
    assert result == sample

    # ModelAdapter reports unloaded state and raises error when predict is attempted before loading
    adapter = SignModelAdapter()
    assert adapter.is_loaded() is False
    with pytest.raises((NotImplementedError, RuntimeError)):
        adapter.predict()
