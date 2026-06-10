"""PolicyMind-AI — FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import graph, health, ingest, query, root
from app.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start-up and shut-down logic."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "PolicyMind-AI starting",
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        graph_provider=settings.graph_provider,
        vector_store=settings.vector_store_provider,
    )
    yield
    logger.info("PolicyMind-AI shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "GraphRAG-powered policy document intelligence platform. "
            "Ingest policy PDFs, extract knowledge graphs, and ask "
            "citation-backed questions over complex policy corpora."
        ),
        contact={
            "name": "PolicyMind-AI",
            "url": "https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────
    PREFIX = "/api/v1"
    app.include_router(root.router,   tags=["Root"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix=PREFIX, tags=["Ingestion"])
    app.include_router(query.router,  prefix=PREFIX, tags=["Query"])
    app.include_router(graph.router,  prefix=PREFIX, tags=["Graph"])

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
