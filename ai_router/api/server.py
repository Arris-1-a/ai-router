"""
FastAPI server exposing ai-router as a REST API.

Endpoints:
  - POST /v1/chat/completions — OpenAI-compatible chat completion
  - POST /v1/embeddings — Generate embeddings
  - POST /v1/rag/search — RAG search endpoint
  - GET  /v1/health — Health check
  - GET  /v1/metrics — Metrics and statistics
  - POST /v1/agent/run — Run an agent
  - GET  /v1/providers — List available providers
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    stream: bool = Field(default=False, description="Enable streaming")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    provider: Optional[str] = Field(default=None, description="Preferred provider")
    strategy: Optional[str] = Field(default=None, description="Routing strategy")


class ChatCompletionChoice(BaseModel):
    """Completion choice."""

    index: int = 0
    message: Message
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    provider: Optional[str] = None
    latency_ms: Optional[float] = None
    cost: Optional[float] = None


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    input: List[str] = Field(..., description="Texts to embed")
    model: str = Field(default="text-embedding-3-small", description="Embedding model")
    provider: Optional[str] = Field(default=None)


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class RAGSearchRequest(BaseModel):
    """RAG search request."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results")
    mode: str = Field(default="hybrid_rrf", description="Retrieval mode")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class RAGSearchResult(BaseModel):
    """Single RAG search result."""

    chunk_id: str
    text: str
    score: float
    rank: int


class RAGSearchResponse(BaseModel):
    """RAG search response."""

    query: str
    results: List[RAGSearchResult]
    total_candidates: int
    latency_ms: float


class AgentRunRequest(BaseModel):
    """Agent execution request."""

    task: str = Field(..., description="Task description")
    agent_name: str = Field(default="default", description="Agent name")
    max_steps: int = Field(default=10, description="Maximum steps")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class AgentRunResponse(BaseModel):
    """Agent execution response."""

    task: str
    final_answer: str
    steps: int
    success: bool
    total_latency_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    uptime_seconds: float
    providers: List[str]
    cache_stats: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────
# App State
# ──────────────────────────────────────────────────────────────────


class AppState:
    """Application-wide shared state."""

    def __init__(self):
        self.start_time = time.time()
        self.version = "0.1.0"
        self.providers: Dict[str, Any] = {}
        self.router: Any = None
        self.retriever: Any = None
        self.agents: Dict[str, Any] = {}
        self.cache_manager: Any = None
        self.metrics_tracker: Any = None

    @property
    def uptime(self) -> float:
        """Uptime in seconds."""
        return time.time() - self.start_time


# ──────────────────────────────────────────────────────────────────
# App Factory
# ──────────────────────────────────────────────────────────────────


def create_app(state: Optional[AppState] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        state: Optional AppState instance.

    Returns:
        Configured FastAPI app.
    """
    app_state = state or AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events."""
        # Startup
        print("🚀 ai-router API starting...")
        yield
        # Shutdown
        print("👋 ai-router API shutting down...")

    app = FastAPI(
        title="ai-router API",
        description="Smart LLM API Router — intelligent routing, RAG, and agent system",
        version=app_state.version,
        lifespan=lifespan,
    )

    # Attach state
    app.state.app_state = app_state

    # ── Health ────────────────────────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            version=app_state.version,
            uptime_seconds=app_state.uptime,
            providers=list(app_state.providers.keys()),
            cache_stats=(
                app_state.cache_manager.stats()
                if app_state.cache_manager
                else {}
            ),
        )

    # ── Chat Completions ──────────────────────────────────────────

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint with routing."""
        start_time = time.time()

        try:
            # Route the request if router is configured
            provider_name = request.provider or "openai"
            model = request.model

            provider = app_state.providers.get(provider_name)
            if provider is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider '{provider_name}' not available. "
                            f"Available: {list(app_state.providers.keys())}",
                )

            # Build request
            from ai_router.router.provider import ChatMessage, CompletionRequest

            messages = [
                ChatMessage(role=m.role, content=m.content)
                for m in request.messages
            ]

            completion_req = CompletionRequest(
                messages=messages,
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            # Execute
            response = await provider.complete(completion_req)

            latency = (time.time() - start_time) * 1000

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(
                            role=response.role,
                            content=response.content or "",
                        ),
                        finish_reason=response.finish_reason,
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    total_tokens=response.usage.get("total_tokens", 0),
                ),
                provider=provider_name,
                latency_ms=latency,
                cost=response.cost,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Embeddings ────────────────────────────────────────────────

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def embeddings(request: EmbeddingRequest):
        """Generate embeddings."""
        try:
            provider_name = request.provider or "openai"
            provider = app_state.providers.get(provider_name)

            if provider is None:
                raise HTTPException(status_code=400, detail=f"Provider not found: {provider_name}")

            from ai_router.router.provider import EmbeddingRequest as EmbReq

            emb_req = EmbReq(texts=request.input, model=request.model)
            response = await provider.embed(emb_req)

            data = [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(response.embeddings)
            ]

            return EmbeddingResponse(
                data=data,
                model=request.model,
                usage=UsageInfo(total_tokens=len(request.input)),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── RAG Search ────────────────────────────────────────────────

    @app.post("/v1/rag/search", response_model=RAGSearchResponse)
    async def rag_search(request: RAGSearchRequest):
        """Search the RAG index."""
        if app_state.retriever is None:
            raise HTTPException(status_code=503, detail="RAG retriever not configured")

        try:
            from ai_router.rag.retriever import RetrievalMode

            mode = RetrievalMode(request.mode) if request.mode else RetrievalMode.HYBRID_RRF
            response = await app_state.retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                mode=mode,
                filters=request.filters,
            )

            results = [
                RAGSearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    rank=r.rank,
                )
                for r in response.results
            ]

            return RAGSearchResponse(
                query=request.query,
                results=results,
                total_candidates=response.total_candidates,
                latency_ms=response.latency_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Agent Run ─────────────────────────────────────────────────

    @app.post("/v1/agent/run", response_model=AgentRunResponse)
    async def agent_run(request: AgentRunRequest):
        """Run an agent on a task."""
        agent = app_state.agents.get(request.agent_name)

        if agent is None:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{request.agent_name}' not found. "
                        f"Available: {list(app_state.agents.keys())}",
            )

        try:
            response = await agent.run(
                task=request.task,
                context=request.context,
                max_steps=request.max_steps,
            )

            return AgentRunResponse(
                task=request.task,
                final_answer=response.final_answer,
                steps=len(response.steps),
                success=response.success,
                total_latency_ms=response.total_latency_ms,
                error=response.error,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Providers ─────────────────────────────────────────────────

    @app.get("/v1/providers")
    async def list_providers():
        """List available providers."""
        return {
            "providers": list(app_state.providers.keys()),
            "count": len(app_state.providers),
        }

    # ── Metrics ───────────────────────────────────────────────────

    @app.get("/v1/metrics")
    async def get_metrics():
        """Get system metrics."""
        metrics = {}

        if app_state.metrics_tracker:
            metrics["tracker"] = app_state.metrics_tracker.get_global_stats()

        if app_state.cache_manager:
            metrics["cache"] = app_state.cache_manager.stats()

        metrics["uptime_seconds"] = app_state.uptime

        return metrics

    return app


# ──────────────────────────────────────────────────────────────────
# Server Runner
# ──────────────────────────────────────────────────────────────────


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 4,
) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        reload: Enable auto-reload.
        workers: Number of worker processes.
    """
    import uvicorn

    uvicorn.run(
        "ai_router.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True,
    )
