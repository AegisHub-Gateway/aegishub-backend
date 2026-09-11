# AegisHub Sign Language API

AegisHub Sign Language recognition service built with FastAPI.

## Phase 4: Model Adapter & Mock Inference

This phase implements the Model Adapter layer decoupling the FastAPI service and business logic from Scroll's future LSTM/GRU model. It provides a configurable mock model for development and integration testing.

### Project Structure

```text
app/
├── __init__.py
├── main.py
├── config.py
├── schemas.py
├── errors.py
├── preprocessing.py
├── model_adapter.py
└── sign_service.py

tests/
├── test_health.py
├── test_preprocessing.py
└── test_model_adapter.py

requirements.txt
README.md
.gitignore
.env
```

### Setup & Installation

1. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Run the Tests:**
   ```bash
   pytest -v --tb=short
   ```

5. **Run Linting (CI Check):**
   ```bash
   ruff check .
   ```

### API Endpoints

* **Health Check:** `GET /healthz`
  ```bash
  curl http://localhost:8000/healthz
  ```
  Response:
  ```json
  {"status": "ok", "service": "aegishub-sign-api", "model_version": "not-loaded", "model_loaded": false}
  ```

* **Sign Classification:** `POST /v1/sign/classify`
  ```bash
  curl -X POST http://localhost:8000/v1/sign/classify \
    -H "Content-Type: application/json" \
    -d '{
      "frames": [
        {
          "left_hand": [{"x": 0.12, "y": 0.35, "z": -0.02}],
          "right_hand": [{"x": 0.18, "y": 0.31, "z": -0.01}]
        }
      ]
    }'
  ```
  Response:
  ```json
  {
    "gloss": "none",
    "confidence": 0.0,
    "alternatives": [],
    "below_threshold": true
  }
  ```

---

## Model Adapter Layer (`app/model_adapter.py`)

The adapter abstracts inference models behind a unified, decoupled interface:

* `load(model_path: str | None = None) -> bool`: Loads the model backend (mock or real weights).
* `is_loaded() -> bool`: Returns `True` if the model backend is ready for inference.
* `predict(input_data: Any, **kwargs: Any) -> dict`: Runs inference and evaluates confidence against the configured threshold.
* `get_model_version() -> str`: Returns the active model version identifier.

### Development Mock Model

When `MOCK_MODEL_ENABLED=true`, `SignModelAdapter` uses `MockSignModel`. By default, it returns:

```json
{
  "gloss": "none",
  "confidence": 0.0,
  "alternatives": [],
  "below_threshold": true
}
```

> **Note:** This mock model is exclusively for local service development and contract validation. It does not reflect real AI model predictions or accuracy.

### Configuration (`app/config.py`)

All model parameters are driven by environment variables or `.env`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_PATH` | `None` | Path to production weights (e.g. `models/sign_lstm.pt`). |
| `MODEL_VERSION` | `not-loaded` | Model version tag reported by health and adapter. |
| `MODEL_LOADED` | `false` | Whether the model initializes as loaded. |
| `CONFIDENCE_THRESHOLD` | `0.6` | Confidence cutoff below which `below_threshold` is set to `true`. |
| `CLASS_LABELS` | `["none"]` | Configurable list of sign classification labels. |
| `MOCK_MODEL_ENABLED` | `true` | Enables safe mock inference during development. |

### Placeholder for Scroll's Real Model

The adapter includes `ScrollModelPlaceholder`, safely handling non-existent weight files without server crashes and without inventing:
* Model tensor input shapes
* Class orders or label indices
* Model file formats or names
* Preprocessing hyperparameters
* PyTorch versions

---

## Landmark Preprocessing Pipeline

The preprocessing layer ([`app/preprocessing.py`](file:///home/soterika/Documents/aegishub-backend/app/preprocessing.py)) converts incoming landmark sequences into structured numerical arrays:

* **Wrist Centering:** Shifts hand coordinates so landmark index 0 is at approximately `[0.0, 0.0, 0.0]`.
* **Scale Normalization:** Normalizes coordinates by the maximum Euclidean distance from the wrist to eliminate camera distance bias.
* **Safe Fallback:** Guarantees zero division cannot occur for zero-sized or missing hands.
* **Missing Hands:** Handled via `MissingHandPolicy.ZEROS` (cleanly zero-filled matrix `(21, 3)`).
* **Shapes Supported:** Configurable between 4D `[30, 2, 21, 3]` and 2D flattened `[30, 126]`.
