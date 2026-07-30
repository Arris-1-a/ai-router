FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY ai_router/ ./ai_router/

# Install dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Run the API server
CMD ["ai-router", "serve", "start", "--host", "0.0.0.0", "--port", "8000"]
