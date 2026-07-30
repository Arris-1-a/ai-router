"""
FastAPI middleware for ai-router.

Provides:
  - Request logging with structured output
  - Rate limiting (token bucket)
  - Request ID tracking
  - Latency measurement
  - Error handling and JSON error responses
  - CORS configuration
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


# ──────────────────────────────────────────────────────────────────
# Request ID Middleware
# ──────────────────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Adds a unique request ID to every request.

    Sets X-Request-ID header on the response and makes it available
    via request.state.request_id.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and add request ID.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response with X-Request-ID header.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ──────────────────────────────────────────────────────────────────
# Logging Middleware
# ──────────────────────────────────────────────────────────────────


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging."""

    def __init__(
        self,
        app,
        log_body: bool = False,
        max_body_length: int = 1000,
        exclude_paths: Optional[list] = None,
    ):
        """Initialize logging middleware.

        Args:
            app: FastAPI application.
            log_body: Whether to log request/response bodies.
            max_body_length: Max body length to log.
            exclude_paths: Paths to exclude from logging.
        """
        super().__init__(app)
        self.log_body = log_body
        self.max_body_length = max_body_length
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log request and response.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response.
        """
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        }

        if self.log_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                body_str = body.decode()[:self.max_body_length]
                log_data["body"] = body_str
            except Exception:
                log_data["body"] = "<unable to read>"

        print(f"📥 REQ [{request_id}] {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)},
            )

        # Log response
        latency_ms = (time.time() - start_time) * 1000
        level = "❌" if status_code >= 500 else ("⚠️" if status_code >= 400 else "✅")

        print(
            f"{level} RES [{request_id}] {status_code} "
            f"{latency_ms:.0f}ms {request.url.path}"
        )

        if error:
            print(f"   Error: {error}")

        return response


# ──────────────────────────────────────────────────────────────────
# Rate Limit Middleware
# ──────────────────────────────────────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiting middleware."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 100,
        per_ip: bool = True,
        exclude_paths: Optional[list] = None,
    ):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application.
            requests_per_minute: Max requests per minute.
            burst_size: Burst capacity.
            per_ip: Apply per-IP limits.
            exclude_paths: Paths to exclude from rate limiting.
        """
        super().__init__(app)
        self.rate = requests_per_minute / 60.0
        self.capacity = burst_size
        self.per_ip = per_ip
        self.exclude_paths = exclude_paths or ["/health"]
        self._buckets: Dict[str, Dict[str, float]] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check rate limit before processing.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response or 429 Too Many Requests.
        """
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        key = self._get_key(request)

        if not self._check_and_consume(key):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": "Rate limit exceeded. Please retry later.",
                    "retry_after": int(1 / self.rate),
                },
                headers={"Retry-After": str(int(1 / self.rate))},
            )

        return await call_next(request)

    def _get_key(self, request: Request) -> str:
        """Get the rate limit key for a request.

        Args:
            request: Incoming request.

        Returns:
            Rate limit key.
        """
        if self.per_ip and request.client:
            return f"ip:{request.client.host}"
        return "global"

    def _check_and_consume(self, key: str) -> bool:
        """Check if request is within rate limit.

        Args:
            key: Rate limit key.

        Returns:
            True if allowed.
        """
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = {"tokens": self.capacity, "last": now}

        bucket = self._buckets[key]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False


# ──────────────────────────────────────────────────────────────────
# Timing Middleware
# ──────────────────────────────────────────────────────────────────


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time header to responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Measure and add processing time.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response with X-Process-Time header.
        """
        start = time.time()
        response = await call_next(request)
        process_time = (time.time() - start) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        return response


# ──────────────────────────────────────────────────────────────────
# Error Handler Middleware
# ──────────────────────────────────────────────────────────────────


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handler that returns JSON errors."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Catch and format errors.

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response or JSON error.
        """
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": str(e),
                    "request_id": request_id,
                },
            )


# ──────────────────────────────────────────────────────────────────
# Middleware Setup
# ──────────────────────────────────────────────────────────────────


def setup_middleware(
    app: FastAPI,
    cors_origins: Optional[list] = None,
    enable_logging: bool = True,
    enable_rate_limit: bool = False,
    rate_limit_rpm: int = 60,
) -> FastAPI:
    """Configure all middleware on a FastAPI app.

    Args:
        app: FastAPI application instance.
        cors_origins: CORS allowed origins.
        enable_logging: Enable request logging.
        enable_rate_limit: Enable rate limiting.
        rate_limit_rpm: Rate limit requests per minute.

    Returns:
        The configured FastAPI app.
    """
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID (first, so all others can use it)
    app.add_middleware(RequestIDMiddleware)

    # Timing
    app.add_middleware(TimingMiddleware)

    # Logging
    if enable_logging:
        app.add_middleware(LoggingMiddleware)

    # Rate limiting
    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rate_limit_rpm,
        )

    # Error handler (last, catches everything)
    app.add_middleware(ErrorHandlerMiddleware)

    return app
