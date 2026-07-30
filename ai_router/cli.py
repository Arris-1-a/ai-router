"""
Command-line interface for ai-router.

Usage:
    ai-router serve          Start the API server
    ai-router chat           Interactive chat with routing
    ai-router benchmark      Run benchmarks
    ai-router eval           Evaluate outputs
    ai-router rag            RAG operations
    ai-router agent          Run an agent
    ai-router config         Manage configuration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="ai-router",
    help="Smart LLM API Router CLI",
    add_completion=True,
)

console = Console()

# Subcommand groups
serve_app = typer.Typer(help="Start the API server")
app.add_typer(serve_app, name="serve")

chat_app = typer.Typer(help="Interactive chat")
app.add_typer(chat_app, name="chat")

bench_app = typer.Typer(help="Run benchmarks")
app.add_typer(bench_app, name="benchmark")

eval_app = typer.Typer(help="Evaluate outputs")
app.add_typer(eval_app, name="eval")

rag_app = typer.Typer(help="RAG operations")
app.add_typer(rag_app, name="rag")

agent_app = typer.Typer(help="Agent operations")
app.add_typer(agent_app, name="agent")


# ──────────────────────────────────────────────────────────────────
# Global options
# ──────────────────────────────────────────────────────────────────


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
):
    """ai-router: Smart LLM API Router — intelligent routing for AI providers."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


# ──────────────────────────────────────────────────────────────────
# Serve
# ──────────────────────────────────────────────────────────────────


@serve_app.command("start")
def serve_start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
):
    """Start the API server."""
    console.print(Panel.fit(
        f"[bold green]🚀 Starting ai-router API server[/bold green]\n"
        f"Host: {host}\nPort: {port}\nWorkers: {workers}",
        title="Server",
    ))

    from ai_router.api.server import run_server
    run_server(host=host, port=port, reload=reload, workers=workers)


# ──────────────────────────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────────────────────────


@chat_app.command("send")
def chat_send(
    message: str = typer.Argument(..., help="Message to send"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to use"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider to use"),
    strategy: str = typer.Option("round_robin", "--strategy", "-s", help="Routing strategy"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature"),
):
    """Send a single chat message."""
    console.print(f"[bold]You:[/bold] {message}")

    async def _send():
        from ai_router.router.provider import (
            ChatMessage,
            CompletionRequest,
            create_provider,
        )

        prov = create_provider(provider)
        request = CompletionRequest(
            messages=[ChatMessage(role="user", content=message)],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = await prov.complete(request)

        console.print(f"\n[bold green]AI ({model}):[/bold green]")
        console.print(response.content or "(empty response)")
        console.print(
            f"\n[dim]Latency: {response.latency_ms:.0f}ms | "
            f"Tokens: {response.usage.get('total_tokens', 0)} | "
            f"Cost: ${response.cost:.6f}[/dim]"
        )

    asyncio.run(_send())


@chat_app.command("interactive")
def chat_interactive(
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to use"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider to use"),
    system_prompt: str = typer.Option(
        "You are a helpful assistant.", "--system", help="System prompt"
    ),
):
    """Start interactive chat session."""
    console.print(Panel.fit(
        f"[bold]Interactive Chat[/bold]\n"
        f"Provider: {provider}\n"
        f"Model: {model}\n"
        f"Type 'exit' or 'quit' to end.",
        title="Chat Session",
    ))

    from ai_router.router.provider import ChatMessage, CompletionRequest, create_provider

    prov = create_provider(provider)
    history = [ChatMessage(role="system", content=system_prompt)]

    async def chat_loop():
        while True:
            try:
                user_input = console.input("\n[bold]You:[/bold] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            history.append(ChatMessage(role="user", content=user_input))

            request = CompletionRequest(
                messages=list(history),
                model=model,
                max_tokens=1024,
                temperature=0.7,
            )

            response = await prov.complete(request)

            assistant_msg = response.content or ""
            console.print(f"\n[bold green]AI:[/bold green] {assistant_msg}")
            history.append(ChatMessage(role="assistant", content=assistant_msg))

            # Trim history if too long
            if len(history) > 20:
                history = [history[0]] + history[-19:]

    asyncio.run(chat_loop())


# ──────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────


@bench_app.command("run")
def benchmark_run(
    requests: int = typer.Option(100, "--requests", "-n", help="Number of requests"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider to test"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to test"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Run a performance benchmark."""
    console.print(f"[bold]Running benchmark: {requests} requests @ {concurrency} concurrency[/bold]")

    from ai_router.eval.benchmark import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        name=f"bench_{provider}_{model}",
        num_requests=requests,
        concurrency=concurrency,
        warmup_requests=5,
        request_template={
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )

    async def _run():
        runner = BenchmarkRunner(config=config)
        result = await runner.run()

        # Print report
        report = runner.generate_report(result)
        console.print(report)

        if output:
            runner.export_json(result, output)
            console.print(f"[green]Results saved to: {output}[/green]")

    asyncio.run(_run())


@bench_app.command("compare")
def benchmark_compare(
    providers: str = typer.Option(
        "openai,deepseek,google", "--providers", help="Comma-separated providers"
    ),
    requests: int = typer.Option(50, "--requests", "-n", help="Requests per provider"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrency"),
):
    """Compare performance across providers."""
    provider_list = [p.strip() for p in providers.split(",")]

    console.print(f"[bold]Comparing providers: {provider_list}[/bold]")

    from ai_router.eval.benchmark import BenchmarkConfig, BenchmarkRunner

    configs = []
    for prov in provider_list:
        configs.append(BenchmarkConfig(
            name=prov,
            num_requests=requests,
            concurrency=concurrency,
            request_template={
                "provider": prov,
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ))

    async def _run():
        runner = BenchmarkRunner()
        results = await runner.compare(configs)
        report = runner.compare_report(results)
        console.print(report)

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────
# Eval
# ──────────────────────────────────────────────────────────────────


@eval_app.command("score")
def eval_score(
    candidate: str = typer.Argument(..., help="Candidate text"),
    reference: str = typer.Argument(..., help="Reference text"),
    metrics: str = typer.Option(
        "bleu,rouge,f1", "--metrics", "-m", help="Comma-separated metrics"
    ),
):
    """Score a candidate against a reference."""
    from ai_router.eval.scorer import Scorer, ScorerType

    scorer = Scorer()
    metric_list = [ScorerType(m.strip()) for m in metrics.split(",")]

    result = scorer.score(candidate, reference, metrics=metric_list)

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Details", style="dim")

    for name, sr in result.scores.items():
        details = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sr.details.items()
            if k not in ("reason",)
        )
        table.add_row(name, f"{sr.score:.4f}", details[:80])

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────────────────────────


@rag_app.command("chunk")
def rag_chunk(
    file: str = typer.Argument(..., help="File to chunk"),
    strategy: str = typer.Option("recursive", "--strategy", "-s", help="Chunk strategy"),
    chunk_size: int = typer.Option(512, "--size", help="Chunk size"),
    overlap: int = typer.Option(50, "--overlap", help="Chunk overlap"),
):
    """Chunk a text file."""
    from ai_router.rag.chunker import ChunkStrategy, create_chunker

    with open(file, "r") as f:
        text = f.read()

    strat = ChunkStrategy(strategy)
    chunker = create_chunker(strategy=strat, chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk(text)

    console.print(f"[bold]File: {file}[/bold]")
    console.print(f"Strategy: {strategy}, Size: {chunk_size}, Overlap: {overlap}")
    console.print(f"Chunks: {len(chunks)}")

    table = Table(title="Chunks")
    table.add_column("#", style="cyan")
    table.add_column("Tokens", style="yellow")
    table.add_column("Preview", style="white")

    for chunk in chunks[:20]:
        table.add_row(
            str(chunk.index),
            str(chunk.token_count),
            chunk.text[:80].replace("\n", " ") + "...",
        )

    console.print(table)


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="Search query"),
    directory: str = typer.Option(".", "--dir", "-d", help="Directory with documents"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Search documents using RAG."""
    import glob

    console.print(f"[bold]Searching: {query}[/bold] in {directory}")

    # Load documents
    docs = []
    for filepath in glob.glob(f"{directory}/**/*.txt", recursive=True)[:50]:
        with open(filepath, "r") as f:
            docs.append(f.read())

    if not docs:
        console.print("[red]No .txt files found in directory[/red]")
        return

    async def _search():
        from ai_router.rag.retriever import HybridRetriever

        retriever = HybridRetriever(embedder=None, top_k=top_k)
        await retriever.index_documents(docs)
        result = await retriever.retrieve(query, top_k=top_k)

        table = Table(title=f"Search Results: {query}")
        table.add_column("#", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Preview", style="white")

        for r in result.results:
            table.add_row(
                str(r.rank + 1),
                f"{r.score:.4f}",
                r.text[:100].replace("\n", " ") + "...",
            )

        console.print(table)
        console.print(f"[dim]{result.latency_ms:.1f}ms[/dim]")

    asyncio.run(_search())


# ──────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────


@agent_app.command("run")
def agent_run(
    task: str = typer.Argument(..., help="Task for the agent"),
    max_steps: int = typer.Option(10, "--max-steps", help="Maximum steps"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run an agent on a task."""
    console.print(f"[bold]Agent Task:[/bold] {task}")

    from ai_router.agents.base import Agent, AgentConfig
    from ai_router.agents.tool import ToolRegistry, calculator, current_time, web_search_simulator

    # Create a simple agent
    registry = ToolRegistry()
    registry.register_from_function(calculator)
    registry.register_from_function(current_time)
    registry.register_from_function(web_search_simulator)

    tools_dict = {
        name: {
            "func": registry.get(name).func if registry.get(name) else None,
            "description": registry.get(name).description if registry.get(name) else "",
            "parameters": {},
        }
        for name in ["calculator", "current_time", "web_search_simulator"]
    }

    config = AgentConfig(
        name="CLI Agent",
        description="A command-line AI assistant",
        max_steps=max_steps,
        verbose=verbose,
    )

    # Simple agent with basic ReAct parsing
    class SimpleAgent(Agent):
        def _parse_response(self, response: str) -> dict:
            import re
            result = {}
            thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|$)', response, re.DOTALL)
            action_match = re.search(r'Action:\s*(\S+)', response)
            action_input_match = re.search(r'Action Input:\s*(.+?)(?=\n(?:Thought|Observation|$)|$)', response, re.DOTALL)
            final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)

            if thought_match:
                result["thought"] = thought_match.group(1).strip()
            if final_match:
                result["final_answer"] = final_match.group(1).strip()
            if action_match:
                result["action"] = action_match.group(1).strip()
            if action_input_match:
                input_str = action_input_match.group(1).strip()
                try:
                    result["action_input"] = json.loads(input_str)
                except json.JSONDecodeError:
                    result["action_input"] = {"input": input_str}

            return result

    agent = SimpleAgent(config=config, tools=tools_dict)

    async def _run():
        response = await agent.run(task)
        console.print(f"\n[bold green]Final Answer:[/bold green] {response.final_answer}")
        console.print(f"\n[dim]Steps: {len(response.steps)} | "
                      f"Latency: {response.total_latency_ms:.0f}ms | "
                      f"Success: {response.success}[/dim]")

    asyncio.run(_run())


@agent_app.command("list-tools")
def agent_list_tools():
    """List available built-in tools."""
    from ai_router.agents.tool import ToolRegistry, calculator, current_time, web_search_simulator, text_length

    registry = ToolRegistry()
    for fn in [calculator, current_time, web_search_simulator, text_length]:
        registry.register_from_function(fn)

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Description", style="white")

    for tool_def in registry.list_all():
        table.add_row(
            tool_def.name,
            tool_def.category.value,
            tool_def.description[:80],
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────


@app.command("config")
def show_config():
    """Show current configuration."""
    table = Table(title="ai-router Configuration")

    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Environment variables
    env_vars = {
        "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else "(not set)",
        "ANTHROPIC_API_KEY": "***" if os.getenv("ANTHROPIC_API_KEY") else "(not set)",
        "DEEPSEEK_API_KEY": "***" if os.getenv("DEEPSEEK_API_KEY") else "(not set)",
        "GOOGLE_API_KEY": "***" if os.getenv("GOOGLE_API_KEY") else "(not set)",
    }

    for key, value in env_vars.items():
        table.add_row(key, value, "env")

    # Version
    from ai_router import __version__
    table.add_row("version", __version__, "package")

    console.print(table)


@app.command("version")
def show_version():
    """Show version information."""
    from ai_router import __version__
    console.print(f"[bold]ai-router[/bold] v{__version__}")
    console.print("[dim]Smart LLM API Router[/dim]")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────


def main_cli():
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    main_cli()
