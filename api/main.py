"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import SessionMiddleware
from api.session_store import cleanup_expired
from api.routers import data_source, profiling, statistics, visualizations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Periodic cleanup of expired sessions."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            cleanup_expired()

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Analytics Engine API",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS - allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookie middleware
app.add_middleware(SessionMiddleware)

# Register routers
app.include_router(data_source.router)
app.include_router(profiling.router)
app.include_router(statistics.router)
app.include_router(visualizations.router)


@app.get("/health")
def health():
    return {"status": "healthy"}
