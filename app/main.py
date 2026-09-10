"""
AegisHub Gateway — FastAPI Application Entrypoint
==================================================
Wires together CORS, health checks, and the versioned API surface for
all three inference modalities: sign-language recognition, derma-scan
triage, and lip-reading.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import derma, lipread, sign

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aegishub")

_START_TIME = time.time()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
)

# --- CORS ---
# Wide open for hackathon/local development so the Next.js frontend can
# call the API from any origin/port without preflight friction. Note:
# `allow_credentials` is left False here — with `allow_origins=["*"]`,
# browsers reject wildcard-origin responses that also carry credentials,
# and this API is stateless (JSON/file bodies only, no cookies), so no
# credentialed requests are expected. Tighten `CORS_ORIGINS` to explicit
# domains before any production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(sign.router, prefix=settings.API_V1_PREFIX)
app.include_router(derma.router, prefix=settings.API_V1_PREFIX)
app.include_router(lipread.router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Meta"], summary="API root")
async def root() -> dict:
    """Basic service metadata and quick links, useful for a smoke test."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/health", tags=["Meta"], summary="Health check")
async def health_check() -> dict:
    """Returns system status and process uptime for uptime monitors / load balancers."""
    uptime_seconds = round(time.time() - _START_TIME, 2)
    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
