# ai-router

<p align="center">
  <b>Smart LLM API Router</b><br>
  Intelligent routing · Load balancing · Fallback · Cost optimization<br>
  智能路由 · 负载均衡 · 故障转移 · 成本优化
</p>

---

## Overview | 概述

**ai-router** is a production-ready Python toolkit for intelligent LLM API management. It provides a unified interface for routing requests across multiple AI providers (OpenAI, Anthropic, DeepSeek, Google, and more) with automatic load balancing, fallback, cost optimization, and comprehensive metrics.

**ai-router** 是一个生产级的 Python 工具包，用于智能管理 LLM API 请求。它提供统一接口，支持跨多个 AI 提供商（OpenAI、Anthropic、DeepSeek、Google 等）的智能路由，具备自动负载均衡、故障转移、成本优化和全面的指标监控。

## Architecture | 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        ai-router                            │
├───────────┬───────────┬──────────┬──────────┬──────────────┤
│  Router   │    RAG    │  Agents  │   Eval   │     API      │
│           │           │          │          │              │
│ Provider  │ Chunker   │  Base    │Benchmark │  FastAPI      │
│ Strategy  │ Embedder  │  Tool    │ Scorer   │  Middleware   │
│ Cache     │ Retriever │Orchestr. │          │              │
│ Fallback  │ Reranker  │ Memory   │          │              │
│ Metrics   │           │          │          │              │
│RateLimit  │           │          │          │              │
└───────────┴───────────┴──────────┴──────────┴──────────────┘
```

## Features | 功能特性

### 🧠 Smart Router | 智能路由器
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, Google Gemini, extensible
- **Strategies**: Round-robin, weighted, lowest-latency, lowest-cost, semantic, adaptive
- **Caching**: LRU + semantic similarity dedup cache
- **Fallback**: Circuit breaker, automatic retry, multi-level fallback chains
- **Rate Limiting**: Token bucket algorithm, per-provider & global limits
- **Metrics**: Latency, cost, success rate tracking with alerts

### 📚 RAG Pipeline | RAG 管线
- **Chunking**: Fixed-size, sentence, paragraph, recursive, markdown, sliding window
- **Embedding**: OpenAI, sentence-transformers, extensible backends
- **Retrieval**: Hybrid (BM25 + vector), RRF fusion, weighted combination
- **Reranking**: Score-based, MMR diversity, cross-encoder, LLM judge

### 🤖 Agent Framework | Agent 框架
- **ReAct Agent**: Reasoning + Acting loop with tool use
- **Tool System**: Decorator-based registration, JSON Schema, validation
- **Orchestration**: Sequential, parallel, debate, manager-worker patterns
- **Memory**: Short-term, long-term, episodic, working memory

### 📊 Evaluation | 评估
- **Scoring**: BLEU, ROUGE-1/2/L, semantic similarity, F1, exact match
- **Benchmarking**: Latency, throughput, cost, success rate under load

### 🌐 API Server | API 服务
- **FastAPI**: OpenAI-compatible `/v1/chat/completions`, embeddings, RAG, agent endpoints
- **Middleware**: Logging, rate limiting, request ID, timing, CORS

## Quick Start | 快速开始

### Installation | 安装

```bash
pip install ai-router
```

Or from source:

```bash
git clone https://github.com/Arris-1-a/ai-router.git
cd ai-router
pip install -e ".[dev]"
```

### Basic Usage | 基本用法

```python
import asyncio
from ai_router import create_provider, ChatMessage, CompletionRequest

async def main():
    # Create a provider
    provider = create_provider("openai", api_key="your-key")

    # Send a completion request
    request = CompletionRequest(
        messages=[ChatMessage(role="user", content="Hello, how are you?")],
        model="gpt-4o-mini",
    )

    response = await provider.complete(request)
    print(response.content)

asyncio.run(main())
```

### Routing with Strategy | 策略路由

```python
from ai_router import (
    RouteTarget, RouteRequest, RoundRobinStrategy,
    PerformanceTracker
)

targets = [
    RouteTarget(provider="openai", model="gpt-4o-mini", weight=3.0),
    RouteTarget(provider="deepseek", model="deepseek-chat", weight=2.0),
    RouteTarget(provider="anthropic", model="claude-3-haiku", weight=1.0),
]

strategy = RoundRobinStrategy()
request = RouteRequest(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)

decision = strategy.select(targets, request)
print(f"Routing to: {decision.target.provider}:{decision.target.model}")
```

### RAG Pipeline | RAG 管线

```python
from ai_router.rag import Chunker, HybridRetriever, RetrievalMode

# Chunk documents
chunker = Chunker(strategy="recursive", chunk_size=512)
chunks = chunker.chunk(your_document)

# Index and retrieve
retriever = HybridRetriever(mode=RetrievalMode.HYBRID_RRF)
await retriever.index_documents([c.text for c in chunks])
results = await retriever.retrieve("your query", top_k=5)
```

### CLI Usage | 命令行使用

```bash
# Start API server
ai-router serve start --port 8000

# Interactive chat
ai-router chat interactive --provider openai --model gpt-4o-mini

# Run benchmarks
ai-router benchmark run --requests 100 --concurrency 10

# Evaluate outputs
ai-router eval score "candidate text" "reference text" --metrics bleu,rouge

# RAG chunking
ai-router rag chunk document.txt --strategy recursive

# Run an agent
ai-router agent run "What is 15 * 23 + 100?"
```

### API Server | API 服务

```bash
ai-router serve start --host 0.0.0.0 --port 8000
```

Then call the API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration | 配置

Set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

## Development | 开发

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=ai_router --cov-report=html

# Lint
ruff check ai_router/

# Type check
mypy ai_router/
```

## Docker | Docker 部署

```bash
docker build -t ai-router .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  ai-router
```

Or with docker-compose:

```bash
docker-compose up
```

## Project Structure | 项目结构

```
ai-router/
├── ai_router/
│   ├── __init__.py
│   ├── cli.py                 # CLI tool
│   ├── router/                # Core router
│   │   ├── provider.py        # Multi-provider abstraction
│   │   ├── strategy.py        # Routing strategies
│   │   ├── cache.py           # Response caching
│   │   ├── fallback.py        # Fault tolerance
│   │   ├── metrics.py         # Metrics tracking
│   │   └── rate_limiter.py    # Rate limiting
│   ├── rag/                   # RAG pipeline
│   │   ├── chunker.py         # Text chunking
│   │   ├── embedder.py        # Embedding generation
│   │   ├── retriever.py       # Hybrid retrieval
│   │   └── reranker.py        # Re-ranking
│   ├── agents/                # Agent framework
│   │   ├── base.py            # Agent base class
│   │   ├── tool.py            # Tool system
│   │   ├── orchestrator.py    # Multi-agent orchestration
│   │   └── memory.py          # Memory management
│   ├── eval/                  # Evaluation
│   │   ├── benchmark.py       # Performance benchmarks
│   │   └── scorer.py          # Scoring metrics
│   └── api/                   # REST API
│       ├── server.py          # FastAPI server
│       └── middleware.py      # Middleware
├── tests/                     # Test suite
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## License | 许可证

MIT License — see [LICENSE](LICENSE) file.

## Contributing | 贡献

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。
