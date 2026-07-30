"""API server subpackage — FastAPI REST server and middleware."""

from ai_router.api.server import (
    AppState,
    create_app,
    run_server,
)
from ai_router.api.middleware import (
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
    setup_middleware,
)

__all__ = [
    # Server
    "AppState", "create_app", "run_server",
    # Middleware
    "ErrorHandlerMiddleware", "LoggingMiddleware", "RateLimitMiddleware",
    "RequestIDMiddleware", "TimingMiddleware", "setup_middleware",
]
