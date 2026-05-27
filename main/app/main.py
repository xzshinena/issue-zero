"""Issue Zero API — cross-repo issue intelligence search engine."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.db import close_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.ml.classifiers import _try_load_setfit
    _try_load_setfit()
    yield
    close_pool()


app = FastAPI(
    title="Issue Zero API",
    description="Cross-repo issue intelligence search engine",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": str(exc), "status_code": 422},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Mount real routers
# ---------------------------------------------------------------------------

from app.api.routes.search import router as search_router  # noqa: E402
from app.api.routes.ingest import router as ingest_router  # noqa: E402

app.include_router(search_router)
app.include_router(ingest_router)

# ---------------------------------------------------------------------------
# SPA static files (mounted last so API routes take priority)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

_DIST = _Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles  # noqa: E402
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
