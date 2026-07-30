"""
Tool definition and registry system for agents.

Provides:
  - Tool definition with JSON Schema validation
  - Tool registry for organizing and discovering tools
  - Parameter validation and type coercion
  - Async/sync tool execution
  - Tool result formatting
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, get_type_hints


class ToolCategory(str, Enum):
    """Tool category classifications."""

    SEARCH = "search"
    FILE = "file"
    WEB = "web"
    CODE = "code"
    DATA = "data"
    API = "api"
    CALCULATION = "calculation"
    KNOWLEDGE = "knowledge"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class ToolDefinition:
    """Complete tool definition with metadata."""

    name: str
    description: str
    func: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.CUSTOM
    is_async: bool = False
    timeout: float = 30.0
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible JSON Schema.

        Returns:
            JSON Schema dict for the tool.
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a simple dict representation.

        Returns:
            Dict with tool metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "is_async": self.is_async,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "version": self.version,
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    tool_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Tool Decorator
# ──────────────────────────────────────────────────────────────────


def tool(
    name: Optional[str] = None,
    description: str = "",
    category: ToolCategory = ToolCategory.CUSTOM,
    timeout: float = 30.0,
    retry_count: int = 0,
    tags: Optional[List[str]] = None,
) -> Callable:
    """Decorator to create a tool from a function.

    Automatically extracts parameter info from function signature
    and type hints.

    Args:
        name: Tool name (defaults to function name).
        description: Tool description.
        category: Tool category.
        timeout: Execution timeout.
        retry_count: Retry count on failure.
        tags: Tags for organization.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        # Extract parameters
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        params = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, str)
            type_str = "string"
            if param_type in (int, float):
                type_str = "number"
            elif param_type == bool:
                type_str = "boolean"
            elif param_type in (list, tuple, set):
                type_str = "array"
            elif param_type == dict:
                type_str = "object"

            tool_param = ToolParameter(
                name=param_name,
                type=type_str,
                description=f"Parameter: {param_name}",
                required=param.default is inspect.Parameter.empty,
                default=None if param.default is inspect.Parameter.empty else param.default,
            )
            params.append(tool_param)

        is_async = asyncio.iscoroutinefunction(func)

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            func=func,
            parameters=params,
            category=category,
            is_async=is_async,
            timeout=timeout,
            retry_count=retry_count,
            tags=tags or [],
        )

        func._tool_definition = definition  # type: ignore[attr-defined]
        return func

    return decorator


# ──────────────────────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for managing and discovering tools.

    Tools can be registered by category, searched, and executed
    with automatic parameter validation.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[ToolCategory, Set[str]] = {c: set() for c in ToolCategory}
        self._tags_index: Dict[str, Set[str]] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        timeout: float = 30.0,
        retry_count: int = 0,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name.
            func: Callable function.
            description: Tool description.
            parameters: Parameters list.
            category: Tool category.
            timeout: Execution timeout.
            retry_count: Retries on failure.
            tags: Organizational tags.
        """
        if parameters is None:
            # Auto-extract from function signature
            sig = inspect.signature(func)
            hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
            parameters = []
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                ptype = hints.get(pname, str)
                type_str = "string"
                if ptype in (int, float):
                    type_str = "number"
                elif ptype == bool:
                    type_str = "boolean"
                elif ptype in (list, tuple, set):
                    type_str = "array"

                parameters.append(ToolParameter(
                    name=pname,
                    type=type_str,
                    required=param.default is inspect.Parameter.empty,
                ))

        is_async = asyncio.iscoroutinefunction(func)

        definition = ToolDefinition(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
            category=category,
            is_async=is_async,
            timeout=timeout,
            retry_count=retry_count,
            tags=tags or [],
        )

        self._tools[name] = definition
        self._categories[category].add(name)

        for tag in (tags or []):
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(name)

        # Initialize stats
        self._execution_stats[name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_latency": 0.0,
        }

    def register_from_function(self, func: Callable) -> str:
        """Register a tool from a @tool decorated function.

        Args:
            func: Tool-decorated function.

        Returns:
            Tool name.
        """
        if hasattr(func, '_tool_definition'):
            definition = func._tool_definition  # type: ignore[attr-defined]
            self._tools[definition.name] = definition
            self._categories[definition.category].add(definition.name)
            for tag in definition.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = set()
                self._tags_index[tag].add(definition.name)
            self._execution_stats[definition.name] = {
                "calls": 0, "successes": 0, "failures": 0, "total_latency": 0.0
            }
            return definition.name
        raise ValueError("Function must be decorated with @tool")

    def register_many(self, tools: List[Callable]) -> List[str]:
        """Register multiple decorated functions at once.

        Args:
            tools: List of @tool decorated functions.

        Returns:
            List of registered tool names.
        """
        return [self.register_from_function(t) for t in tools]

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name.

        Returns:
            True if tool was removed.
        """
        if name not in self._tools:
            return False
        definition = self._tools.pop(name)
        self._categories[definition.category].discard(name)
        for tag in definition.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(name)
        self._execution_stats.pop(name, None)
        return True

    async def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Execute a registered tool.

        Args:
            name: Tool name.
            params: Parameter dict.
            timeout: Override timeout.

        Returns:
            ToolResult with execution outcome.
        """
        import time

        definition = self._tools.get(name)
        if definition is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )

        params = params or {}
        start = time.monotonic()

        try:
            # Validate parameters
            valid, error = self._validate_params(definition, params)
            if not valid:
                return ToolResult(success=False, data=None, error=error)

            # Execute with timeout
            timeout = timeout or definition.timeout
            if definition.is_async:
                result = await asyncio.wait_for(
                    definition.func(**params),
                    timeout=timeout,
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(definition.func, **params),
                    ),
                    timeout=timeout,
                )

            latency = (time.monotonic() - start) * 1000

            # Update stats
            stats = self._execution_stats[name]
            stats["calls"] += 1
            stats["successes"] += 1
            stats["total_latency"] += latency

            return ToolResult(
                success=True,
                data=result,
                latency_ms=latency,
                tool_name=name,
            )

        except asyncio.TimeoutError:
            self._execution_stats[name]["failures"] += 1
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{name}' timed out after {timeout}s",
                tool_name=name,
            )
        except Exception as e:
            self._execution_stats[name]["failures"] += 1
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=name,
            )

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition.

        Args:
            name: Tool name.

        Returns:
            ToolDefinition or None.
        """
        return self._tools.get(name)

    def get_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category.

        Args:
            category: Tool category.

        Returns:
            List of ToolDefinitions.
        """
        return [self._tools[name] for name in self._categories.get(category, set())]

    def get_by_tag(self, tag: str) -> List[ToolDefinition]:
        """Get all tools with a tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of ToolDefinitions.
        """
        return [self._tools[name] for name in self._tags_index.get(tag, set())]

    def search(self, query: str) -> List[ToolDefinition]:
        """Search tools by name or description.

        Args:
            query: Search query.

        Returns:
            List of matching ToolDefinitions.
        """
        query_lower = query.lower()
        matches = []
        for name, defn in self._tools.items():
            if query_lower in name.lower() or query_lower in defn.description.lower():
                matches.append(defn)
        return matches

    def list_all(self) -> List[ToolDefinition]:
        """List all registered tools.

        Returns:
            List of all ToolDefinitions.
        """
        return list(self._tools.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for all tools.

        Returns:
            Dict with per-tool and aggregate stats.
        """
        total_calls = 0
        total_successes = 0
        total_failures = 0

        for name, stats in self._execution_stats.items():
            total_calls += stats["calls"]
            total_successes += stats["successes"]
            total_failures += stats["failures"]

        return {
            "total_tools": len(self._tools),
            "total_calls": total_calls,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": (
                total_successes / total_calls if total_calls > 0 else 0.0
            ),
            "by_tool": dict(self._execution_stats),
            "categories": {
                cat.value: len(tools) for cat, tools in self._categories.items()
            },
        }

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible format.

        Returns:
            List of OpenAI tool schemas.
        """
        return [t.to_json_schema() for t in self._tools.values()]

    def _validate_params(
        self,
        definition: ToolDefinition,
        params: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate parameters against tool definition.

        Args:
            definition: Tool definition.
            params: Provided parameters.

        Returns:
            Tuple of (is_valid, error_message).
        """
        for param in definition.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

            if param.name in params:
                value = params[param.name]
                # Type coercion
                if param.type == "number" and not isinstance(value, (int, float)):
                    try:
                        params[param.name] = float(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter '{param.name}' must be a number"

                if param.enum and value not in param.enum:
                    return False, (
                        f"Parameter '{param.name}' must be one of {param.enum}"
                    )

        return True, None


# ──────────────────────────────────────────────────────────────────
# Built-in Tools
# ──────────────────────────────────────────────────────────────────


@tool(
    name="calculator",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, abs, sin, cos, log.",
    category=ToolCategory.CALCULATION,
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate.

    Returns:
        Result as string.
    """
    import math

    # Safe evaluation with limited globals
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "int": int,
        "float": float,
    }

    try:
        # Compile for basic safety check
        code = compile(expression, "<calculator>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"'{name}' is not allowed")

        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="current_time",
    description="Get the current date and time in ISO format.",
    category=ToolCategory.SYSTEM,
)
def current_time(timezone_offset: int = 0) -> str:
    """Get current time with optional timezone offset.

    Args:
        timezone_offset: UTC offset in hours (default 0).

    Returns:
        ISO-formatted datetime string.
    """
    from datetime import datetime, timedelta, timezone as tz

    dt = datetime.now(tz.utc) + timedelta(hours=timezone_offset)
    return dt.isoformat()


@tool(
    name="text_length",
    description="Count characters, words, and lines in text.",
    category=ToolCategory.CALCULATION,
)
def text_length(text: str) -> str:
    """Count text statistics.

    Args:
        text: Text to analyze.

    Returns:
        String with character, word, and line counts.
    """
    chars = len(text)
    words = len(text.split())
    lines = text.count("\n") + 1
    return json.dumps({
        "characters": chars,
        "words": words,
        "lines": lines,
    })


@tool(
    name="web_search_simulator",
    description="Simulate a web search (for testing/demo purposes). Returns mock results.",
    category=ToolCategory.SEARCH,
)
def web_search_simulator(query: str, num_results: int = 5) -> str:
    """Simulate web search results for demonstration.

    Args:
        query: Search query.
        num_results: Number of results.

    Returns:
        JSON string with mock search results.
    """
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Result {i+1} for: {query}",
            "url": f"https://example.com/result-{i+1}",
            "snippet": f"This is a simulated search result about {query}. "
                       f"It demonstrates how the agent handles search tool output.",
            "relevance": round(1.0 - i * 0.1, 1),
        })
    return json.dumps(results, indent=2)
