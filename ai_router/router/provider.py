"""
Multi-provider abstraction layer.

Provides a unified interface for interacting with multiple LLM providers
including OpenAI, Anthropic, DeepSeek, and Google (Gemini).
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ──────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────


class ModelCapability(str, Enum):
    """Capabilities a model may support."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


class ProviderType(str, Enum):
    """Supported provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a specific model."""

    model_id: str
    provider: ProviderType
    capabilities: List[ModelCapability] = field(default_factory=lambda: [ModelCapability.CHAT])
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    context_window: int = 8192
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system", "user", "assistant", "function", "tool"
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class CompletionRequest:
    """Standardized completion request across all providers."""

    messages: List[ChatMessage]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Standardized completion response across all providers."""

    content: Optional[str] = None
    role: str = "assistant"
    finish_reason: str = "stop"
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    latency_ms: float = 0.0
    cost: float = 0.0
    provider: Optional[ProviderType] = None
    raw_response: Optional[Any] = None


@dataclass
class EmbeddingRequest:
    """Standardized embedding request."""

    texts: List[str]
    model: str = "text-embedding-3-small"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Standardized embedding response."""

    embeddings: List[List[float]]
    model: str = ""
    dimensions: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0


# ──────────────────────────────────────────────────────────────────
# Provider exceptions
# ──────────────────────────────────────────────────────────────────


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class AuthenticationError(ProviderError):
    """Invalid API key or authentication failure."""

    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    pass


class InvalidRequestError(ProviderError):
    """Invalid request parameters."""

    pass


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    pass


class ContextLengthExceededError(ProviderError):
    """Input exceeds the model's context window."""

    pass


# ──────────────────────────────────────────────────────────────────
# Base Provider
# ──────────────────────────────────────────────────────────────────


class Provider(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement the core completion and embedding methods.
    Provides common retry logic, error handling, and cost calculation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        organization: Optional[str] = None,
        default_model: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key or os.getenv(self._api_key_env())
        self.base_url = base_url or self._default_base_url()
        self.timeout = timeout
        self.max_retries = max_retries
        self.organization = organization
        self.default_model = default_model or self._default_model()
        self._http_client = http_client
        self._initialized = False

    # ── Abstract methods ──────────────────────────────────────────

    @abstractmethod
    def _api_key_env(self) -> str:
        """Return the environment variable name for the API key."""
        ...

    @abstractmethod
    def _default_base_url(self) -> str:
        """Return the default API base URL."""
        ...

    @abstractmethod
    def _default_model(self) -> str:
        """Return the default model identifier."""
        ...

    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type enum value."""
        ...

    @abstractmethod
    async def _complete_impl(self, request: CompletionRequest) -> CompletionResponse:
        """Provider-specific completion implementation."""
        ...

    @abstractmethod
    async def _stream_impl(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """Provider-specific streaming implementation."""
        ...

    @abstractmethod
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for the given model and token counts."""
        ...

    @abstractmethod
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        ...

    # ── HTTP client management ────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None or not self._initialized:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._default_headers(),
            )
            self._initialized = True
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._initialized = False

    def _default_headers(self) -> Dict[str, str]:
        """Build default request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers[self._org_header_name()] = self.organization
        return headers

    def _org_header_name(self) -> str:
        """Organization header name (overridable)."""
        return "OpenAI-Organization"

    # ── Public API ────────────────────────────────────────────────

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a completion request with retry logic.

        Args:
            request: Standardized CompletionRequest.

        Returns:
            CompletionResponse with the model output and metadata.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limited.
            ProviderTimeoutError: If the request times out.
            ProviderError: For other provider failures.
        """
        start_time = time.monotonic()
        try:
            response = await self._retry_complete(request)
            response.latency_ms = (time.monotonic() - start_time) * 1000
            response.provider = self.provider_type()
            return response
        except Exception as e:
            self._classify_and_raise(e)

    async def _retry_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute completion with retry logic."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (ProviderTimeoutError, RateLimitError, httpx.NetworkError)
            ),
            reraise=True,
        )
        async def _do_complete():
            return await self._complete_impl(request)

        return await _do_complete()

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """Stream a completion response.

        Args:
            request: Standardized CompletionRequest with stream=True.

        Yields:
            Chunks of CompletionResponse as they arrive.
        """
        request.stream = True
        start_time = time.monotonic()
        try:
            async for chunk in self._stream_impl(request):
                chunk.latency_ms = (time.monotonic() - start_time) * 1000
                chunk.provider = self.provider_type()
                yield chunk
        except Exception as e:
            self._classify_and_raise(e)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given texts.

        Args:
            request: Standardized EmbeddingRequest.

        Returns:
            EmbeddingResponse with the generated embeddings.
        """
        start_time = time.monotonic()
        try:
            response = await self._embed_impl(request)
            response.latency_ms = (time.monotonic() - start_time) * 1000
            return response
        except Exception as e:
            self._classify_and_raise(e)

    async def _embed_impl(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Default embedding implementation — override for better performance."""
        raise NotImplementedError(
            f"Embedding not supported by {self.provider_type().value}"
        )

    def _classify_and_raise(self, exc: Exception) -> None:
        """Classify an exception into a ProviderError subtype."""
        if isinstance(exc, ProviderError):
            raise exc
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
            raise AuthenticationError(str(exc)) from exc
        if "429" in msg or "rate limit" in msg:
            raise RateLimitError(str(exc)) from exc
        if "timeout" in msg or "timed out" in msg:
            raise ProviderTimeoutError(str(exc)) from exc
        if "context_length" in msg or "maximum context" in msg:
            raise ContextLengthExceededError(str(exc)) from exc
        if "400" in msg:
            raise InvalidRequestError(str(exc)) from exc
        raise ProviderError(str(exc)) from exc

    # ── Health check ──────────────────────────────────────────────

    async def health_check(self) -> Tuple[bool, float]:
        """Check provider health and latency.

        Returns:
            Tuple of (is_healthy, latency_in_ms).
        """
        start = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/v1/models", timeout=10.0)
            latency = (time.monotonic() - start) * 1000
            return response.status_code < 500, latency
        except Exception:
            return False, (time.monotonic() - start) * 1000

    def supports_model(self, model_id: str) -> bool:
        """Check if this provider supports the given model."""
        return model_id in self._supported_models()

    def _supported_models(self) -> List[str]:
        """Return list of supported model IDs."""
        return []


# ──────────────────────────────────────────────────────────────────
# OpenAI Provider
# ──────────────────────────────────────────────────────────────────


class OpenAIProvider(Provider):
    """OpenAI API provider implementation."""

    def _api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    def _default_base_url(self) -> str:
        return "https://api.openai.com"

    def _default_model(self) -> str:
        return "gpt-4o-mini"

    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate OpenAI cost based on model pricing.

        Prices as of 2024-07. Update as needed.
        """
        pricing = {
            "gpt-4o": (5.0, 15.0),          # $5/$15 per 1M tokens
            "gpt-4o-mini": (0.15, 0.60),    # $0.15/$0.60 per 1M tokens
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-4": (30.0, 60.0),
            "gpt-3.5-turbo": (0.50, 1.50),
            "text-embedding-3-small": (0.02, 0.02),
            "text-embedding-3-large": (0.13, 0.13),
        }
        input_price, output_price = pricing.get(model, (1.0, 3.0))
        cost = (prompt_tokens / 1_000_000) * input_price
        cost += (completion_tokens / 1_000_000) * output_price
        return cost

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get model information for OpenAI models."""
        model_map = {
            "gpt-4o": ModelInfo(
                model_id="gpt-4o",
                provider=ProviderType.OPENAI,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                    ModelCapability.JSON_MODE,
                ],
                max_tokens=4096,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                context_window=128000,
                supports_vision=True,
                supports_function_calling=True,
            ),
            "gpt-4o-mini": ModelInfo(
                model_id="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                    ModelCapability.JSON_MODE,
                ],
                max_tokens=4096,
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                context_window=128000,
                supports_function_calling=True,
            ),
        }
        return model_map.get(model_id, ModelInfo(model_id=model_id, provider=ProviderType.OPENAI))

    async def _complete_impl(self, request: CompletionRequest) -> CompletionResponse:
        """OpenAI completion implementation using their SDK-style format over HTTP."""
        client = await self._get_client()

        payload = self._build_openai_payload(request)
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        cost = self._calculate_cost(
            model=data.get("model", request.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

        return CompletionResponse(
            content=message.get("content", ""),
            role=message.get("role", "assistant"),
            finish_reason=choice.get("finish_reason", "stop"),
            model=data.get("model", request.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            function_call=message.get("function_call"),
            tool_calls=message.get("tool_calls"),
            cost=cost,
            raw_response=data,
        )

    def _build_openai_payload(self, request: CompletionRequest) -> Dict[str, Any]:
        """Build OpenAI-specific request payload."""
        messages = []
        for msg in request.messages:
            message: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                message["name"] = msg.name
            if msg.function_call:
                message["function_call"] = msg.function_call
            if msg.tool_calls:
                message["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                message["tool_call_id"] = msg.tool_call_id
            messages.append(message)

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }

        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.functions:
            payload["functions"] = request.functions
        if request.function_call:
            payload["function_call"] = request.function_call
        if request.response_format:
            payload["response_format"] = request.response_format

        return payload

    async def _stream_impl(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """OpenAI streaming implementation."""
        client = await self._get_client()
        payload = self._build_openai_payload(request)
        payload["stream"] = True

        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        yield CompletionResponse(
                            content=delta.get("content", ""),
                            role=delta.get("role", "assistant"),
                            finish_reason=choice.get("finish_reason"),
                            model=data.get("model", request.model),
                            function_call=delta.get("function_call"),
                            tool_calls=delta.get("tool_calls"),
                            raw_response=data,
                        )
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def _embed_impl(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """OpenAI embedding implementation."""
        client = await self._get_client()
        payload = {
            "model": request.model,
            "input": request.texts,
        }
        response = await client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", request.model),
            dimensions=len(embeddings[0]) if embeddings else 0,
            cost=self._calculate_cost(request.model, 0, 0),
        )

    def _supported_models(self) -> List[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]


# ──────────────────────────────────────────────────────────────────
# Anthropic Provider
# ──────────────────────────────────────────────────────────────────


class AnthropicProvider(Provider):
    """Anthropic (Claude) API provider implementation."""

    def _api_key_env(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _default_model(self) -> str:
        return "claude-3-5-sonnet-20240620"

    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        headers["anthropic-version"] = "2023-06-01"
        headers["x-api-key"] = self.api_key or ""
        if "Authorization" in headers:
            del headers["Authorization"]
        return headers

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Anthropic pricing (per 1M tokens)."""
        pricing = {
            "claude-3-5-sonnet-20240620": (3.0, 15.0),
            "claude-3-opus-20240229": (15.0, 75.0),
            "claude-3-sonnet-20240229": (3.0, 15.0),
            "claude-3-haiku-20240307": (0.25, 1.25),
        }
        input_price, output_price = pricing.get(model, (3.0, 15.0))
        cost = (prompt_tokens / 1_000_000) * input_price
        cost += (completion_tokens / 1_000_000) * output_price
        return cost

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get model info for Anthropic models."""
        model_map = {
            "claude-3-5-sonnet-20240620": ModelInfo(
                model_id="claude-3-5-sonnet-20240620",
                provider=ProviderType.ANTHROPIC,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                ],
                max_tokens=4096,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                context_window=200000,
                supports_vision=True,
            ),
        }
        return model_map.get(
            model_id, ModelInfo(model_id=model_id, provider=ProviderType.ANTHROPIC)
        )

    async def _complete_impl(self, request: CompletionRequest) -> CompletionResponse:
        """Anthropic Messages API completion."""
        client = await self._get_client()

        system_prompt = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content if isinstance(msg.content, str) else msg.content,
                })

        payload: Dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature >= 0:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop

        response = await client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return CompletionResponse(
            content=content,
            role="assistant",
            finish_reason=data.get("stop_reason", "end_turn"),
            model=data.get("model", request.model),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            cost=self._calculate_cost(
                data.get("model", request.model), prompt_tokens, completion_tokens
            ),
            raw_response=data,
        )

    async def _stream_impl(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """Anthropic streaming implementation."""
        import json

        client = await self._get_client()

        system_prompt = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                messages.append({"role": msg.role, "content": msg.content})

        payload: Dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature >= 0:
            payload["temperature"] = request.temperature

        async with client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            yield CompletionResponse(
                                content=delta.get("text", ""),
                                role="assistant",
                                raw_response=event,
                            )
                        elif event.get("type") == "message_stop":
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _supported_models(self) -> List[str]:
        return [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]


# ──────────────────────────────────────────────────────────────────
# DeepSeek Provider
# ──────────────────────────────────────────────────────────────────


class DeepSeekProvider(Provider):
    """DeepSeek API provider — OpenAI-compatible interface."""

    def _api_key_env(self) -> str:
        return "DEEPSEEK_API_KEY"

    def _default_base_url(self) -> str:
        return "https://api.deepseek.com"

    def _default_model(self) -> str:
        return "deepseek-chat"

    def provider_type(self) -> ProviderType:
        return ProviderType.DEEPSEEK

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """DeepSeek pricing (per 1M tokens)."""
        pricing = {
            "deepseek-chat": (0.14, 0.28),
            "deepseek-reasoner": (0.14, 0.28),
            "deepseek-coder": (0.14, 0.28),
        }
        input_price, output_price = pricing.get(model, (0.14, 0.28))
        cost = (prompt_tokens / 1_000_000) * input_price
        cost += (completion_tokens / 1_000_000) * output_price
        return cost

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get model info for DeepSeek models."""
        model_map = {
            "deepseek-chat": ModelInfo(
                model_id="deepseek-chat",
                provider=ProviderType.DEEPSEEK,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
                max_tokens=4096,
                cost_per_1k_input=0.00014,
                cost_per_1k_output=0.00028,
                context_window=65536,
            ),
            "deepseek-coder": ModelInfo(
                model_id="deepseek-coder",
                provider=ProviderType.DEEPSEEK,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
                max_tokens=4096,
                cost_per_1k_input=0.00014,
                cost_per_1k_output=0.00028,
                context_window=65536,
            ),
        }
        return model_map.get(
            model_id, ModelInfo(model_id=model_id, provider=ProviderType.DEEPSEEK)
        )

    async def _complete_impl(self, request: CompletionRequest) -> CompletionResponse:
        """DeepSeek completion — OpenAI-compatible format."""
        client = await self._get_client()

        messages = []
        for msg in request.messages:
            entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            messages.append(entry)

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = request.stop

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return CompletionResponse(
            content=message.get("content", ""),
            role=message.get("role", "assistant"),
            finish_reason=choice.get("finish_reason", "stop"),
            model=data.get("model", request.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            cost=self._calculate_cost(
                data.get("model", request.model),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            ),
            raw_response=data,
        )

    async def _stream_impl(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """DeepSeek streaming — OpenAI-compatible SSE format."""
        import json

        client = await self._get_client()

        messages = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        yield CompletionResponse(
                            content=delta.get("content", ""),
                            role=delta.get("role", "assistant"),
                            finish_reason=choice.get("finish_reason"),
                            model=data.get("model", request.model),
                            raw_response=data,
                        )
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def _supported_models(self) -> List[str]:
        return ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]


# ──────────────────────────────────────────────────────────────────
# Google (Gemini) Provider
# ──────────────────────────────────────────────────────────────────


class GoogleProvider(Provider):
    """Google Generative AI (Gemini) provider implementation."""

    def _api_key_env(self) -> str:
        return "GOOGLE_API_KEY"

    def _default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com"

    def _default_model(self) -> str:
        return "gemini-1.5-pro"

    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Gemini pricing (per 1M tokens)."""
        pricing = {
            "gemini-1.5-pro": (3.50, 10.50),
            "gemini-1.5-flash": (0.075, 0.30),
            "gemini-1.0-pro": (0.50, 1.50),
        }
        input_price, output_price = pricing.get(model, (3.50, 10.50))
        cost = (prompt_tokens / 1_000_000) * input_price
        cost += (completion_tokens / 1_000_000) * output_price
        return cost

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get model info for Gemini models."""
        model_map = {
            "gemini-1.5-pro": ModelInfo(
                model_id="gemini-1.5-pro",
                provider=ProviderType.GOOGLE,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                    ModelCapability.FUNCTION_CALLING,
                ],
                max_tokens=8192,
                cost_per_1k_input=0.0035,
                cost_per_1k_output=0.0105,
                context_window=1048576,
                supports_vision=True,
                supports_function_calling=True,
            ),
            "gemini-1.5-flash": ModelInfo(
                model_id="gemini-1.5-flash",
                provider=ProviderType.GOOGLE,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING,
                    ModelCapability.FUNCTION_CALLING,
                ],
                max_tokens=8192,
                cost_per_1k_input=0.000075,
                cost_per_1k_output=0.0003,
                context_window=1048576,
                supports_vision=True,
                supports_function_calling=True,
            ),
        }
        return model_map.get(
            model_id, ModelInfo(model_id=model_id, provider=ProviderType.GOOGLE)
        )

    async def _complete_impl(self, request: CompletionRequest) -> CompletionResponse:
        """Gemini completion implementation."""
        client = await self._get_client()

        contents = self._messages_to_gemini_contents(request.messages)

        system_instruction = None
        for msg in request.messages:
            if msg.role == "system":
                system_instruction = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        generation_config: Dict[str, Any] = {
            "maxOutputTokens": request.max_tokens,
        }
        if request.temperature >= 0:
            generation_config["temperature"] = request.temperature
        if request.top_p < 1.0:
            generation_config["topP"] = request.top_p
        if request.stop:
            generation_config["stopSequences"] = request.stop

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        url = f"/v1beta/models/{request.model}:generateContent"
        if self.api_key:
            url += f"?key={self.api_key}"

        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderError("No candidates returned from Gemini")

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in content_parts)

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)

        return CompletionResponse(
            content=text,
            role="assistant",
            finish_reason=candidate.get("finishReason", "STOP"),
            model=request.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            cost=self._calculate_cost(request.model, prompt_tokens, completion_tokens),
            raw_response=data,
        )

    async def _stream_impl(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """Gemini streaming implementation."""
        import json

        client = await self._get_client()

        contents = self._messages_to_gemini_contents(request.messages)

        generation_config: Dict[str, Any] = {
            "maxOutputTokens": request.max_tokens,
        }
        if request.temperature >= 0:
            generation_config["temperature"] = request.temperature

        body = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        url = f"/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
        if self.api_key:
            url += f"&key={self.api_key}"

        async with client.stream("POST", url, json=body) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts)
                            yield CompletionResponse(
                                content=text,
                                role="assistant",
                                raw_response=data,
                            )
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def _messages_to_gemini_contents(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage list to Gemini contents format."""
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue  # System handled separately
            role = "user" if msg.role == "user" else "model"
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict):
                        parts.append(part)
                    else:
                        parts.append({"text": str(part)})
            contents.append({"role": role, "parts": parts})
        return contents

    def _supported_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]


# ──────────────────────────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────────────────────────


_PROVIDER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "google": GoogleProvider,
}


def register_provider(name: str, provider_cls: type) -> None:
    """Register a custom provider class.

    Args:
        name: Unique provider name.
        provider_cls: Provider subclass to register.
    """
    _PROVIDER_REGISTRY[name] = provider_cls


def get_provider_class(name: str) -> Optional[type]:
    """Get a registered provider class by name.

    Args:
        name: Provider name.

    Returns:
        Provider class or None if not found.
    """
    return _PROVIDER_REGISTRY.get(name)


def list_providers() -> List[str]:
    """List all registered provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def create_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Provider:
    """Factory function to create a provider instance.

    Args:
        provider_type: Provider name (openai, anthropic, deepseek, google).
        api_key: Optional API key override.
        base_url: Optional base URL override.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A Provider instance.

    Raises:
        ValueError: If the provider type is unknown.
    """
    cls = get_provider_class(provider_type)
    if cls is None:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Available: {list_providers()}"
        )
    return cls(api_key=api_key, base_url=base_url, **kwargs)
