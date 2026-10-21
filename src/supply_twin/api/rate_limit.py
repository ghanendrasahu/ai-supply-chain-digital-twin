"""Per-client fixed-window rate limiting (in-process; guards a single worker).

Limits are read from settings on every request so tests can retune them via
environment variables and ``get_settings().cache_clear()``.
"""

from __future__ import annotations

import time as _time
from collections import defaultdict
from typing import ClassVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


def _clock() -> float:
    """Indirection so tests can advance time without patching the stdlib clock."""
    return _time.time()


class FixedWindowRateLimiter(BaseHTTPMiddleware):
    """Cap requests per client at ``rate_limit_max_requests`` per ``rate_limit_window``."""

    _instances: ClassVar[list[FixedWindowRateLimiter]] = []

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, list[float]] = defaultdict(list)
        FixedWindowRateLimiter._instances.append(self)

    def _limits(self) -> tuple[int, int]:
        from supply_twin.config import get_settings

        settings = get_settings()
        return settings.rate_limit_window, settings.rate_limit_max_requests

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        window, max_requests = self._limits()
        if max_requests <= 0:  # limiter disabled
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = _clock()
        active = [t for t in self._hits[client] if now - t < window]
        self._hits[client] = active

        if len(active) >= max_requests:
            retry_after = max(1, int(window - (now - active[0]))) if active else window
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )

        self._hits[client].append(now)
        return await call_next(request)
