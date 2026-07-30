"""
Multi-agent orchestration framework.

Enables coordination between multiple specialized agents:
  - Sequential task delegation
  - Parallel agent execution
  - Debate/conversation between agents
  - Hierarchical agent structures (manager → workers)
  - Shared context and memory
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ai_router.agents.base import Agent, AgentConfig, AgentResponse, AgentState


class OrchestrationMode(str, Enum):
    """Orchestration modes."""

    SEQUENTIAL = "sequential"     # One agent at a time, in order
    PARALLEL = "parallel"         # All agents run simultaneously
    DEBATE = "debate"             # Agents discuss and converge
    MANAGER_WORKER = "manager_worker"  # Manager delegates to workers
    ROUTER = "router"             # Route to best-fit agent


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL
    max_parallel: int = 5
    timeout_per_agent: float = 300.0
    max_debate_rounds: int = 3
    consensus_threshold: float = 0.7  # For debate mode
    verbose: bool = False


@dataclass
class AgentTask:
    """A task assigned to a specific agent."""

    agent_name: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    """Complete result of an orchestration run."""

    success: bool = True
    results: Dict[str, AgentResponse] = field(default_factory=dict)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Agent Orchestrator
# ──────────────────────────────────────────────────────────────────


class AgentOrchestrator:
    """Orchestrates multiple agents to solve complex tasks.

    Supports various coordination patterns:
    - Sequential: chain agents one after another
    - Parallel: run agents concurrently
    - Debate: agents discuss and reach consensus
    - Manager-Worker: hierarchical delegation
    """

    def __init__(
        self,
        agents: Optional[Dict[str, Agent]] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """Initialize the orchestrator.

        Args:
            agents: Dictionary of named agents.
            config: Orchestrator configuration.
        """
        self.agents = agents or {}
        self.config = config or OrchestratorConfig()

    # ── Agent Management ──────────────────────────────────────────

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the pool.

        Args:
            name: Unique agent name.
            agent: Agent instance.
        """
        self.agents[name] = agent

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the pool.

        Args:
            name: Agent name.

        Returns:
            True if removed.
        """
        return self.agents.pop(name, None) is not None

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent or None.
        """
        return self.agents.get(name)

    # ── Orchestration ─────────────────────────────────────────────

    async def orchestrate(
        self,
        tasks: List[AgentTask],
        mode: Optional[OrchestrationMode] = None,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Execute tasks across agents.

        Args:
            tasks: List of AgentTask objects.
            mode: Override orchestration mode.
            shared_context: Context shared across all agents.

        Returns:
            OrchestrationResult with all agent outputs.
        """
        mode = mode or self.config.mode
        start_time = time.monotonic()

        if mode == OrchestrationMode.SEQUENTIAL:
            result = await self._run_sequential(tasks, shared_context)
        elif mode == OrchestrationMode.PARALLEL:
            result = await self._run_parallel(tasks, shared_context)
        elif mode == OrchestrationMode.DEBATE:
            result = await self._run_debate(tasks, shared_context)
        elif mode == OrchestrationMode.MANAGER_WORKER:
            result = await self._run_manager_worker(tasks, shared_context)
        elif mode == OrchestrationMode.ROUTER:
            result = await self._run_router(tasks, shared_context)
        else:
            result = OrchestrationResult(
                success=False,
                errors={"orchestrator": f"Unknown mode: {mode}"},
            )

        result.total_latency_ms = (time.monotonic() - start_time) * 1000

        # Compile final answer
        if result.success and not result.final_answer:
            result.final_answer = self._compile_final_answer(result)

        return result

    async def _run_sequential(
        self,
        tasks: List[AgentTask],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Run tasks sequentially, passing context between them.

        Each agent's output becomes part of the next agent's context.

        Args:
            tasks: Ordered task list.
            shared_context: Shared context dict.

        Returns:
            OrchestrationResult.
        """
        result = OrchestrationResult()
        context = shared_context or {}
        accumulated_output = ""

        for task in tasks:
            agent = self.agents.get(task.agent_name)
            if agent is None:
                result.errors[task.agent_name] = f"Agent '{task.agent_name}' not found"
                result.success = False
                continue

            # Inject context from previous agents
            task_context = {**context, **task.context}
            if accumulated_output:
                task_context["previous_output"] = accumulated_output
                full_task = (
                    f"{task.task}\n\nPrevious agent output:\n{accumulated_output}"
                )
            else:
                full_task = task.task

            if self.config.verbose:
                print(f"\n🤖 Running agent: {task.agent_name}")
                print(f"   Task: {full_task[:200]}...")

            try:
                response = await asyncio.wait_for(
                    agent.run(full_task, context=task_context),
                    timeout=self.config.timeout_per_agent,
                )
                result.results[task.agent_name] = response
                result.total_tokens += response.total_tokens
                result.total_cost += response.total_cost
                result.execution_order.append(task.agent_name)
                accumulated_output = response.final_answer
            except asyncio.TimeoutError:
                result.errors[task.agent_name] = "Timeout"
                result.success = False
            except Exception as e:
                result.errors[task.agent_name] = str(e)
                result.success = False

        return result

    async def _run_parallel(
        self,
        tasks: List[AgentTask],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Run tasks in parallel, collecting results.

        Args:
            tasks: Task list.
            shared_context: Shared context.

        Returns:
            OrchestrationResult.
        """
        result = OrchestrationResult()
        context = shared_context or {}

        async def run_one(task: AgentTask) -> Tuple[str, Optional[AgentResponse], Optional[str]]:
            agent = self.agents.get(task.agent_name)
            if agent is None:
                return task.agent_name, None, f"Agent '{task.agent_name}' not found"

            try:
                response = await asyncio.wait_for(
                    agent.run(task.task, context={**context, **task.context}),
                    timeout=self.config.timeout_per_agent,
                )
                return task.agent_name, response, None
            except asyncio.TimeoutError:
                return task.agent_name, None, "Timeout"
            except Exception as e:
                return task.agent_name, None, str(e)

        # Run within parallel limit
        semaphore = asyncio.Semaphore(self.config.max_parallel)

        async def bounded_run(task: AgentTask) -> Tuple[str, Optional[AgentResponse], Optional[str]]:
            async with semaphore:
                return await run_one(task)

        coros = [bounded_run(t) for t in tasks]
        outcomes = await asyncio.gather(*coros)

        for name, response, error in outcomes:
            result.execution_order.append(name)
            if error:
                result.errors[name] = error
                result.success = False
            elif response is not None:
                result.results[name] = response
                result.total_tokens += response.total_tokens
                result.total_cost += response.total_cost

        return result

    async def _run_debate(
        self,
        tasks: List[AgentTask],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Run agents in debate mode — they discuss and converge.

        Args:
            tasks: Task list (topic shared across agents).
            shared_context: Shared context.

        Returns:
            OrchestrationResult with consensus.
        """
        result = OrchestrationResult()
        if not tasks:
            return result

        # The first task defines the debate topic
        topic = tasks[0].task
        agents_to_use = [self.agents[t.agent_name] for t in tasks if t.agent_name in self.agents]

        if len(agents_to_use) < 2:
            result.errors["debate"] = "Need at least 2 agents for debate"
            result.success = False
            return result

        # Initial positions
        positions: Dict[str, str] = {}
        for agent in agents_to_use:
            debate_task = f"State your position on the following topic:\n\n{topic}"
            response = await agent.run(debate_task)
            positions[agent.config.name] = response.final_answer
            result.results[agent.config.name] = response
            result.total_tokens += response.total_tokens

        # Debate rounds
        for round_num in range(self.config.max_debate_rounds):
            if self.config.verbose:
                print(f"\n🗣 Debate Round {round_num + 1}")

            new_positions = {}
            for agent in agents_to_use:
                # Show all other positions
                others = {
                    name: pos
                    for name, pos in positions.items()
                    if name != agent.config.name
                }
                others_text = "\n\n".join(
                    f"{name}'s position: {pos}" for name, pos in others.items()
                )
                debate_prompt = (
                    f"Topic: {topic}\n\n"
                    f"Other agents' positions:\n{others_text}\n\n"
                    f"Critically evaluate and refine your position. Respond with "
                    f"your updated position. If you agree with others, say so clearly."
                )

                response = await agent.run(debate_prompt)
                new_positions[agent.config.name] = response.final_answer
                result.execution_order.append(f"{agent.config.name}_round{round_num+1}")

            positions = new_positions

        # Final answer: first agent's last position
        first_name = agents_to_use[0].config.name
        result.final_answer = positions.get(first_name, "No consensus reached.")
        result.metadata["positions"] = positions

        return result

    async def _run_manager_worker(
        self,
        tasks: List[AgentTask],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Manager delegates subtasks to worker agents.

        First task defines the manager, remaining are workers.

        Args:
            tasks: Task list (first=manager, rest=workers).
            shared_context: Shared context.

        Returns:
            OrchestrationResult.
        """
        result = OrchestrationResult()

        if len(tasks) < 2:
            result.errors["manager"] = "Need manager + at least 1 worker"
            result.success = False
            return result

        manager_task = tasks[0]
        worker_tasks = tasks[1:]

        manager = self.agents.get(manager_task.agent_name)
        if manager is None:
            result.errors["manager"] = f"Manager agent not found: {manager_task.agent_name}"
            result.success = False
            return result

        # First, get worker capabilities
        workers_info = []
        for wt in worker_tasks:
            agent = self.agents.get(wt.agent_name)
            if agent:
                workers_info.append({
                    "name": wt.agent_name,
                    "description": agent.config.description,
                    "tools": agent.list_tools(),
                })

        # Manager creates a plan
        plan_prompt = (
            f"Task: {manager_task.task}\n\n"
            f"Available workers:\n"
            + "\n".join(
                f"- {w['name']}: {w['description']}"
                for w in workers_info
            )
            + "\n\nCreate a plan delegating subtasks to workers. "
            "For each worker, specify: Worker name, task description, "
            "and any specific instructions. Format as:\n"
            "Worker: <name>\nTask: <task>\n\n"
            "Then wait for results and synthesize a final answer."
        )

        plan_response = await manager.run(plan_prompt)
        result.results["manager_plan"] = plan_response

        # Execute worker tasks in parallel
        worker_coros = []
        for wt in worker_tasks:
            agent = self.agents.get(wt.agent_name)
            if agent:
                worker_coros.append(self._execute_worker(agent, wt, shared_context))

        worker_results = await asyncio.gather(*worker_coros, return_exceptions=True)

        for wr in worker_results:
            if isinstance(wr, Exception):
                result.errors[str(wr)] = str(wr)
            elif isinstance(wr, tuple):
                name, response = wr
                result.results[name] = response
                result.total_tokens += response.total_tokens
                result.total_cost += response.total_cost
                result.execution_order.append(name)

        # Manager synthesizes
        worker_outputs = "\n\n".join(
            f"Worker {name}: {resp.final_answer}"
            for name, resp in result.results.items()
            if name != "manager_plan"
        )
        synthesis_prompt = (
            f"Original task: {manager_task.task}\n\n"
            f"Worker outputs:\n{worker_outputs}\n\n"
            f"Synthesize a comprehensive final answer from the worker outputs."
        )

        final = await manager.run(synthesis_prompt)
        result.results["manager_final"] = final
        result.final_answer = final.final_answer
        result.total_tokens += final.total_tokens
        result.total_cost += final.total_cost

        return result

    async def _run_router(
        self,
        tasks: List[AgentTask],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Route a task to the most suitable agent.

        Each task includes a 'route_key' in context for matching.

        Args:
            tasks: Task list.
            shared_context: Shared context.

        Returns:
            OrchestrationResult.
        """
        result = OrchestrationResult()

        for task in tasks:
            route_key = task.context.get("route_key", task.agent_name)
            agent = self.agents.get(route_key)

            if agent is None:
                # Try to find best match by description
                agent = self._find_best_agent(task.task)

            if agent is None:
                result.errors[route_key] = "No suitable agent found"
                result.success = False
                continue

            try:
                response = await agent.run(task.task, context=task.context)
                result.results[route_key] = response
                result.total_tokens += response.total_tokens
                result.total_cost += response.total_cost
                result.execution_order.append(route_key)
            except Exception as e:
                result.errors[route_key] = str(e)
                result.success = False

        return result

    async def _execute_worker(
        self,
        agent: Agent,
        task: AgentTask,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, AgentResponse]:
        """Execute a worker agent task.

        Args:
            agent: Agent instance.
            task: Task to execute.
            shared_context: Shared context.

        Returns:
            Tuple of (agent_name, response).
        """
        response = await agent.run(task.task, context={
            **(shared_context or {}),
            **task.context,
        })
        return agent.config.name, response

    def _find_best_agent(self, task: str) -> Optional[Agent]:
        """Find the best agent for a task based on description matching.

        Args:
            task: Task description.

        Returns:
            Best matching agent or None.
        """
        task_lower = task.lower()
        best_score = 0.0
        best_agent = None

        for name, agent in self.agents.items():
            desc = agent.config.description.lower()
            # Simple keyword overlap scoring
            keywords = set(task_lower.split()) & set(desc.split())
            score = len(keywords)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def _compile_final_answer(self, result: OrchestrationResult) -> str:
        """Compile a final answer from all agent results.

        Args:
            result: Orchestration result.

        Returns:
            Compiled final answer string.
        """
        if not result.results:
            return "No results produced."

        if len(result.results) == 1:
            return list(result.results.values())[0].final_answer

        # Concatenate answers with agent labels
        parts = []
        for name, response in result.results.items():
            if response.success:
                parts.append(f"[{name}]: {response.final_answer}")

        return "\n\n".join(parts) if parts else "All agents failed."

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dict with agent counts, modes, etc.
        """
        return {
            "total_agents": len(self.agents),
            "agents": list(self.agents.keys()),
            "mode": self.config.mode.value,
            "modes_available": [m.value for m in OrchestrationMode],
        }
