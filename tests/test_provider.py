"""Tests for provider module."""

import pytest
from ai_router.router.provider import (
    AnthropicProvider,
    ChatMessage,
    CompletionRequest,
    DeepSeekProvider,
    GoogleProvider,
    ModelCapability,
    ModelInfo,
    OpenAIProvider,
    ProviderError,
    ProviderType,
    create_provider,
    get_provider_class,
    list_providers,
    register_provider,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_defaults(self):
        """Test default values for ModelInfo."""
        info = ModelInfo(model_id="test-model", provider=ProviderType.OPENAI)
        assert info.model_id == "test-model"
        assert info.provider == ProviderType.OPENAI
        assert ModelCapability.CHAT in info.capabilities
        assert info.max_tokens == 4096
        assert info.cost_per_1k_input == 0.0
        assert info.supports_streaming is True
        assert info.supports_vision is False
        assert info.context_window == 8192

    def test_model_info_custom(self):
        """Test custom model info."""
        info = ModelInfo(
            model_id="custom",
            provider=ProviderType.OPENAI,
            capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            max_tokens=8192,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            context_window=128000,
            supports_vision=True,
            supports_function_calling=True,
        )
        assert ModelCapability.VISION in info.capabilities
        assert info.max_tokens == 8192
        assert info.supports_vision is True
        assert info.supports_function_calling is True


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_simple_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_message_with_name(self):
        msg = ChatMessage(role="user", content="Hi", name="Alice")
        assert msg.name == "Alice"


class TestCompletionRequest:
    """Tests for CompletionRequest dataclass."""

    def test_default_values(self):
        req = CompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="gpt-4o-mini",
        )
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stream is False

    def test_custom_values(self):
        req = CompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="gpt-4",
            max_tokens=512,
            temperature=0.3,
            stop=["END"],
            stream=True,
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.3
        assert req.stop == ["END"]
        assert req.stream is True


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_type(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_type() == ProviderType.OPENAI

    def test_default_model(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._default_model() == "gpt-4o-mini"

    def test_default_base_url(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._default_base_url() == "https://api.openai.com"

    def test_api_key_env(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._api_key_env() == "OPENAI_API_KEY"

    def test_cost_calculation(self):
        provider = OpenAIProvider(api_key="test-key")
        cost = provider._calculate_cost("gpt-4o-mini", 1000, 500)
        # $0.15/1M input + $0.60/1M output
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 0.0001

    def test_cost_calculation_gpt4(self):
        provider = OpenAIProvider(api_key="test-key")
        cost = provider._calculate_cost("gpt-4o", 1000000, 1000000)
        expected = 5.0 + 15.0  # $5/M input + $15/M output
        assert abs(cost - expected) < 0.01

    def test_model_info(self):
        provider = OpenAIProvider(api_key="test-key")
        info = provider.get_model_info("gpt-4o")
        assert info.model_id == "gpt-4o"
        assert info.supports_vision is True
        assert info.supports_function_calling is True
        assert info.context_window == 128000

    def test_supported_models(self):
        provider = OpenAIProvider(api_key="test-key")
        models = provider._supported_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models

    def test_supports_model(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.supports_model("gpt-4o") is True
        assert provider.supports_model("nonexistent-model") is False


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_type(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.provider_type() == ProviderType.ANTHROPIC

    def test_default_model(self):
        provider = AnthropicProvider(api_key="test-key")
        assert "claude" in provider._default_model()

    def test_cost_calculation(self):
        provider = AnthropicProvider(api_key="test-key")
        cost = provider._calculate_cost("claude-3-haiku-20240307", 1000, 500)
        expected = (1000 / 1_000_000) * 0.25 + (500 / 1_000_000) * 1.25
        assert abs(cost - expected) < 0.0001


class TestDeepSeekProvider:
    """Tests for DeepSeekProvider."""

    def test_provider_type(self):
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.provider_type() == ProviderType.DEEPSEEK

    def test_default_model(self):
        provider = DeepSeekProvider(api_key="test-key")
        assert provider._default_model() == "deepseek-chat"

    def test_cost_is_low(self):
        provider = DeepSeekProvider(api_key="test-key")
        cost = provider._calculate_cost("deepseek-chat", 1000, 500)
        # DeepSeek is very cheap
        assert cost < 0.01

    def test_supported_models(self):
        provider = DeepSeekProvider(api_key="test-key")
        models = provider._supported_models()
        assert "deepseek-chat" in models
        assert "deepseek-coder" in models


class TestGoogleProvider:
    """Tests for GoogleProvider."""

    def test_provider_type(self):
        provider = GoogleProvider(api_key="test-key")
        assert provider.provider_type() == ProviderType.GOOGLE

    def test_default_model(self):
        provider = GoogleProvider(api_key="test-key")
        assert "gemini" in provider._default_model()

    def test_custom_headers(self):
        provider = GoogleProvider(api_key="test-key")
        headers = provider._default_headers()
        assert "x-api-key" in headers
        assert "anthropic-version" in headers


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_list_providers(self):
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "deepseek" in providers
        assert "google" in providers

    def test_get_provider_class(self):
        assert get_provider_class("openai") == OpenAIProvider
        assert get_provider_class("anthropic") == AnthropicProvider
        assert get_provider_class("deepseek") == DeepSeekProvider
        assert get_provider_class("google") == GoogleProvider
        assert get_provider_class("nonexistent") is None

    def test_create_provider(self):
        provider = create_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError):
            create_provider("unknown_provider")

    def test_register_custom_provider(self):
        class CustomProvider(OpenAIProvider):
            pass

        register_provider("custom_test", CustomProvider)
        assert get_provider_class("custom_test") == CustomProvider
        prov = create_provider("custom_test", api_key="x")
        assert isinstance(prov, CustomProvider)


class TestProviderErrors:
    """Tests for provider exception classification."""

    def test_provider_error(self):
        with pytest.raises(ProviderError):
            raise ProviderError("test error")

    def test_error_hierarchy(self):
        from ai_router.router.provider import (
            AuthenticationError,
            InvalidRequestError,
            RateLimitError,
        )
        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(InvalidRequestError, ProviderError)
