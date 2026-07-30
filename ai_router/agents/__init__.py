"""Agent framework subpackage — ReAct agents, tools, memory, orchestration."""

from ai_router.agents.base import (
    Agent,
    AgentConfig,
    AgentResponse,
    AgentState,
    AgentStep,
)
from ai_router.agents.tool import (
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    calculator,
    current_time,
    text_length,
    tool,
    web_search_simulator,
)
from ai_router.agents.memory import (
    AgentMemory,
    ConversationTurn,
    Episode,
    MemoryEntry,
    MemoryType,
)
from ai_router.agents.orchestrator import (
    AgentOrchestrator,
    AgentTask,
    OrchestrationMode,
    OrchestrationResult,
    OrchestratorConfig,
)

__all__ = [
    # Base Agent
    "Agent", "AgentConfig", "AgentResponse", "AgentState", "AgentStep",
    # Tools
    "ToolCategory", "ToolDefinition", "ToolParameter", "ToolRegistry",
    "ToolResult", "calculator", "current_time", "text_length", "tool",
    "web_search_simulator",
    # Memory
    "AgentMemory", "ConversationTurn", "Episode", "MemoryEntry", "MemoryType",
    # Orchestrator
    "AgentOrchestrator", "AgentTask", "OrchestrationMode",
    "OrchestrationResult", "OrchestratorConfig",
]
