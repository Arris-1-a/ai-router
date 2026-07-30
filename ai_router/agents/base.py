"""
ReAct-style Agent base class with tool use and multi-turn reasoning.

Implements:
  - ReAct (Reasoning + Acting) loop
  - Tool calling with structured outputs
  - Conversation memory
  - Streaming support
  - Configurable system prompts
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple, Union


class AgentState(str, Enum):
    """Possible agent states."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""

    step_number: int
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "Assistant"
    description: str = "A helpful AI assistant"
    system_prompt: str = ""
    max_steps: int = 10
    max_tokens_per_step: int = 4096
    temperature: float = 0.7
    model: str = "gpt-4o-mini"
    stop_sequences: List[str] = field(default_factory=lambda: ["\nObservation:"])
    tools: List[str] = field(default_factory=list)
    stream: bool = False
    verbose: bool = False


@dataclass
class AgentResponse:
    """Complete agent response after execution."""

    final_answer: str
    steps: List[AgentStep]
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Base Agent
# ──────────────────────────────────────────────────────────────────


class Agent(ABC):
    """Abstract base class for ReAct agents.

    Implements the core ReAct loop:
    1. Thought: Agent reasons about what to do
    2. Action: Agent executes a tool or responds
    3. Observation: Agent observes the result
    Repeat until finished or max steps reached.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Callable]] = None,
        llm_complete_fn: Optional[Callable] = None,
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration.
            tools: Tool registry mapping tool names to callables.
            llm_complete_fn: Async function for LLM completion.
        """
        self.config = config or AgentConfig()
        self.tools = tools or {}
        self.llm_complete_fn = llm_complete_fn
        self.state = AgentState.IDLE
        self._conversation_history: List[Dict[str, str]] = []
        self._execution_steps: List[AgentStep] = []

    # ── Public API ────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResponse:
        """Execute the agent on a task.

        Args:
            task: The task description or user query.
            context: Optional context variables.
            max_steps: Override max steps.

        Returns:
            AgentResponse with final answer and execution steps.
        """
        start_time = time.monotonic()
        self.state = AgentState.THINKING
        self._execution_steps = []
        max_steps = max_steps or self.config.max_steps

        # Build initial conversation
        self._conversation_history = self._build_initial_messages(task)

        total_tokens = 0
        final_answer = ""
        success = True
        error = None

        try:
            for step_num in range(1, max_steps + 1):
                step_start = time.monotonic()

                # Get LLM response
                step = AgentStep(step_number=step_num)
                response = await self._get_llm_response()

                step.tokens_used = response.get("tokens", 0)
                total_tokens += step.tokens_used

                # Parse the response
                parsed = self._parse_response(response.get("content", ""))

                if parsed.get("final_answer"):
                    step.final_answer = parsed["final_answer"]
                    self._execution_steps.append(step)
                    final_answer = parsed["final_answer"]
                    self.state = AgentState.FINISHED
                    break

                if parsed.get("action"):
                    step.action = parsed["action"]
                    step.action_input = parsed.get("action_input", {})
                    step.thought = parsed.get("thought", "")

                    # Execute the action
                    self.state = AgentState.ACTING
                    observation = await self._execute_action(
                        step.action,
                        step.action_input,
                    )
                    step.observation = observation

                    # Add observation to conversation
                    self._conversation_history.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })

                    self.state = AgentState.OBSERVING
                else:
                    # No action or final answer found — treat as final
                    final_answer = response.get("content", "I'm done.")
                    step.final_answer = final_answer
                    self._execution_steps.append(step)
                    self.state = AgentState.FINISHED
                    break

                step.latency_ms = (time.monotonic() - step_start) * 1000
                self._execution_steps.append(step)

                if self.config.verbose:
                    self._log_step(step)

            if self.state != AgentState.FINISHED:
                final_answer = "I've reached the maximum number of steps."
                self.state = AgentState.FINISHED

        except Exception as e:
            success = False
            error = str(e)
            self.state = AgentState.ERROR
            final_answer = f"Error: {error}"

        total_latency = (time.monotonic() - start_time) * 1000

        return AgentResponse(
            final_answer=final_answer,
            steps=self._execution_steps,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            total_cost=self._estimate_cost(total_tokens),
            success=success,
            error=error,
        )

    async def run_stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run agent with streaming progress updates.

        Args:
            task: Task description.
            context: Optional context.

        Yields:
            Progress events (thought, action, observation, final).
        """
        self._conversation_history = self._build_initial_messages(task)
        max_steps = self.config.max_steps

        for step_num in range(1, max_steps + 1):
            response = await self._get_llm_response()
            parsed = self._parse_response(response.get("content", ""))

            if parsed.get("thought"):
                yield {
                    "type": "thought",
                    "step": step_num,
                    "content": parsed["thought"],
                }

            if parsed.get("final_answer"):
                yield {
                    "type": "final",
                    "step": step_num,
                    "content": parsed["final_answer"],
                }
                return

            if parsed.get("action"):
                action = parsed["action"]
                action_input = parsed.get("action_input", {})

                yield {
                    "type": "action",
                    "step": step_num,
                    "action": action,
                    "input": action_input,
                }

                observation = await self._execute_action(action, action_input)

                yield {
                    "type": "observation",
                    "step": step_num,
                    "content": observation,
                }

                self._conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

        yield {
            "type": "final",
            "step": max_steps,
            "content": "Max steps reached.",
        }

    # ── Tool Management ───────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool for the agent to use.

        Args:
            name: Tool name.
            func: Callable function (sync or async).
            description: Tool description.
            parameters: JSON Schema for parameters.
        """
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters or {},
        }

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the agent.

        Args:
            name: Tool name to remove.
        """
        self.tools.pop(name, None)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool info dicts.
        """
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in self.tools.items()
        ]

    # ── Internal Methods ──────────────────────────────────────────

    async def _get_llm_response(self) -> Dict[str, Any]:
        """Get LLM response for the current conversation.

        Returns:
            Dict with 'content' and 'tokens' keys.

        Raises:
            RuntimeError: If no LLM function is configured.
        """
        if self.llm_complete_fn is None:
            raise RuntimeError(
                "No LLM completion function configured. "
                "Set agent.llm_complete_fn or use a subclass."
            )

        response = await self.llm_complete_fn(
            messages=self._conversation_history,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_step,
            stop=self.config.stop_sequences,
        )

        if isinstance(response, str):
            return {"content": response, "tokens": 0}

        return {
            "content": response.get("content", ""),
            "tokens": response.get("tokens", 0),
        }

    async def _execute_action(
        self,
        action: str,
        action_input: Dict[str, Any],
    ) -> str:
        """Execute a tool action.

        Args:
            action: Tool name.
            action_input: Tool input parameters.

        Returns:
            Observation string.
        """
        tool_info = self.tools.get(action)
        if tool_info is None:
            return f"Error: Unknown action '{action}'. Available tools: {list(self.tools.keys())}"

        try:
            func = tool_info["func"]
            if asyncio.iscoroutinefunction(func):
                result = await func(**action_input)
            else:
                result = func(**action_input)
            return str(result)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def _build_initial_messages(self, task: str) -> List[Dict[str, str]]:
        """Build the initial conversation messages.

        Args:
            task: User task.

        Returns:
            List of message dicts.
        """
        messages = []

        # System prompt
        system = self._build_system_prompt()
        if system:
            messages.append({"role": "system", "content": system})

        # User task
        messages.append({"role": "user", "content": task})

        return messages

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions.

        Returns:
            System prompt string.
        """
        if self.config.system_prompt:
            prompt = self.config.system_prompt
        else:
            prompt = f"You are {self.config.name}, {self.config.description}.\n\n"
            prompt += (
                "You have access to the following tools. Use them to complete tasks.\n\n"
            )

        # Add tool descriptions
        if self.tools:
            prompt += "## Available Tools\n\n"
            for name, info in self.tools.items():
                prompt += f"### {name}\n"
                prompt += f"{info['description']}\n"
                if info.get("parameters"):
                    prompt += f"Parameters: {json.dumps(info['parameters'], indent=2)}\n"
                prompt += "\n"

        # Add output format instructions
        prompt += (
            "## Response Format\n\n"
            "Use the following format for each step:\n\n"
            "Thought: <your reasoning about what to do next>\n"
            "Action: <tool name>\n"
            "Action Input: <JSON parameters for the tool>\n\n"
            "After you have enough information, respond with:\n\n"
            "Thought: I now have the information needed.\n"
            "Final Answer: <your comprehensive answer>\n"
        )

        return prompt

    @abstractmethod
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured parts.

        Args:
            response: Raw LLM response text.

        Returns:
            Dict with 'thought', 'action', 'action_input', 'final_answer' keys.
        """
        ...

    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost based on token usage.

        Args:
            total_tokens: Total tokens used.

        Returns:
            Estimated cost in dollars.
        """
        # Rough estimate: $0.002 per 1K tokens
        return total_tokens * 0.002 / 1000

    def _log_step(self, step: AgentStep) -> None:
        """Log an agent step (verbose mode).

        Args:
            step: The step to log.
        """
        print(f"\n{'='*60}")
        print(f"Step {step.step_number} ({step.latency_ms:.0f}ms)")
        if step.thought:
            print(f"💭 Thought: {step.thought}")
        if step.action:
            print(f"🔧 Action: {step.action}")
            print(f"   Input: {json.dumps(step.action_input, indent=2)}")
        if step.observation:
            print(f"👁 Observation: {step.observation[:200]}")
        if step.final_answer:
            print(f"✅ Final Answer: {step.final_answer[:200]}")
        if step.error:
            print(f"❌ Error: {step.error}")

    def reset(self) -> None:
        """Reset the agent's state and history."""
        self.state = AgentState.IDLE
        self._conversation_history = []
        self._execution_steps = []
