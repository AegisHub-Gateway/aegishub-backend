# AegisHub Gateway — Backend

Multi-modal health accessibility gateway for **GatewayHacks 2026**. The backend follows a **Split-Processing Architecture**: the Next.js frontend does lightweight extraction in-browser (MediaPipe hand/face landmarks, raw media capture) and ships JSON/files to this FastAPI service, which owns all AI inference.

## Modalities

| Modality | Owner | Endpoint | Input | Output |
|---|---|---|---|---|
| Sign Language | Timothy | `POST /api/v1/sign/predict` | JSON — 30 frames × 21 3D hand landmarks | `detected_sign`, `confidence`, `is_emergency` |
| Derma-Scan | Neche | `POST /api/v1/derma/scan` | multipart image upload | `condition`, `urgency`, `confidence`, `summary` |
| Lip-Reading | Neche | `POST /api/v1/lipread/transcribe` | multipart audio + `lip_mesh` JSON string | `transcript`, `confidence`, `is_muffled` |

All three currently run on deterministic **dummy predictors** (see `app/models/model_loader.py`) so the full pipeline works end-to-end without GPU hardware or trained weights. Dropping a `.pt` checkpoint at the configured path (see `app/core/config.py`) transparently switches a model over to real inference — no router or schema changes required.

---

## 1. Setup

```bash
# From the aegishub-backend/ directory
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Torch install note:** `requirements.txt` pins the CPU-only PyTorch wheel to keep hackathon setup fast. If you have a CUDA GPU, install the matching build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead before running `pip install -r requirements.txt`.

## 2. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or via the entrypoint's `__main__` block:

```bash
python -m app.main
```

- Interactive docs (Swagger UI): **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**
- Health check: **http://localhost:8000/health**

## 3. Environment configuration (optional)

All settings in `app/core/config.py` can be overridden via a `.env` file in `aegishub-backend/`:

```dotenv
PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
SIGN_MODEL_PATH=app/models/weights/sign_lstm.pt
DERMA_MODEL_PATH=app/models/weights/derma_mobilenet.pt
```

---

## 4. Testing the endpoints

### Health check

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "uptime_seconds": 12.34, "version": "1.0.0" }
```

### Sign Language — `POST /api/v1/sign/predict`

This endpoint strictly requires **exactly 30 frames**, each with **exactly 21 landmarks**, each coordinate normalized to `[0.0, 1.0]`. Any deviation (missing frames, wrong landmark count, out-of-range coordinates) returns **HTTP 422**.

Generate a valid sample payload with Python:

```bash
python3 - <<'EOF' > sign_payload.json
import json, random

frames = [
    {"landmarks": [
        {"x": round(random.random(), 4), "y": round(random.random(), 4), "z": round(random.random(), 4)}
        for _ in range(21)
    ]}
    for _ in range(30)
]

print(json.dumps({"session_id": "demo-session-001", "frames": frames}))
EOF
```

Then call the endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/sign/predict \
  -H "Content-Type: application/json" \
  -d @sign_payload.json
```

```json
{ "detected_sign": "HELP", "confidence": 0.8734, "is_emergency": true }
```

**Trigger a validation error (422)** — e.g. too few frames:

```bash
curl -X POST http://localhost:8000/api/v1/sign/predict \
  -H "Content-Type: application/json" \
  -d '{"frames": [{"landmarks": []}]}'
```

### Derma-Scan — `POST /api/v1/derma/scan`

```bash
curl -X POST http://localhost:8000/api/v1/derma/scan \
  -F "image=@/path/to/skin_photo.jpg;type=image/jpeg"
```

```json
{
  "condition": "Atypical Mole",
  "urgency": "medium",
  "confidence": 0.8123,
  "summary": "Recommend scheduling a dermatologist consultation within the next 1-2 weeks."
}
```

### Lip-Reading — `POST /api/v1/lipread/transcribe`

`lip_mesh` is a JSON-encoded string form field (not a file) alongside the audio upload:

```bash
curl -X POST http://localhost:8000/api/v1/lipread/transcribe \
  -F "audio=@/path/to/clip.wav;type=audio/wav" \
  -F 'lip_mesh={"frames":[{"points":[{"x":0.1,"y":0.2},{"x":0.15,"y":0.22},{"x":0.2,"y":0.2},{"x":0.15,"y":0.18}]}]}'
```

```json
{ "transcript": "I need help please", "confidence": 0.812, "is_muffled": false }
```

### Testing via Postman

1. **Sign Language**: `POST` request, Body → `raw` → `JSON`, paste `sign_payload.json` contents.
2. **Derma-Scan**: `POST` request, Body → `form-data`, key `image` set to type **File**, select an image.
3. **Lip-Reading**: `POST` request, Body → `form-data`, key `audio` set to type **File**, plus a second key `lip_mesh` set to type **Text** with a JSON string value.

---

## 5. Project structure

```text
aegishub-backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, /health, router mounting
│   ├── core/config.py       # Pydantic BaseSettings (env-driven config)
│   ├── schemas/             # Pydantic v2 request/response contracts
│   │   ├── sign.py
│   │   ├── derma.py
│   │   └── lipread.py
│   ├── routers/             # Endpoint logic per modality
│   │   ├── sign.py
│   │   ├── derma.py
│   │   └── lipread.py
│   └── models/
│       └── model_loader.py  # ModelRegistry — lazy loading + dummy fallbacks
├── requirements.txt
└── .gitignore
```

## 6. Next steps for Scroll (AI/ML)

Drop trained checkpoints into `app/models/weights/`:

- `sign_lstm.pt` — LSTM sign-classification model
- `derma_mobilenet.pt` — MobileNet derma-scan classification model

`ModelRegistry` (`app/models/model_loader.py`) will automatically detect and load them on next process start, replacing the dummy predictors with real inference — no other code changes needed.
