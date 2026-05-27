"""API key authentication dependency.

When API_KEYS is set (comma-separated) in the environment, every protected
route requires an `X-API-Key` header matching one of the configured keys.
When API_KEYS is empty the dependency is a no-op — open access is preserved
for local dev / zero-config deployments.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_header_scheme)) -> str | None:
    """FastAPI dependency — validates X-API-Key when API_KEYS is configured."""
    settings = get_settings()
    raw = (settings.api_keys or "").strip()
    if not raw:
        return None  # open access

    valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not valid_keys:
        return None  # all entries blank → open access

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
