"""
Agent memory management with multiple storage backends.

Supports:
  - Short-term memory (conversation buffer)
  - Long-term memory (persistent key-value store)
  - Episodic memory (past interaction summaries)
  - Semantic memory (knowledge graph / embeddings)
  - Memory summarization and compression
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class MemoryType(str, Enum):
    """Types of memories."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


@dataclass
class MemoryEntry:
    """A single memory entry."""

    key: str
    value: Any
    memory_type: MemoryType = MemoryType.LONG_TERM
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5     # 0-1, how important this memory is
    ttl: Optional[float] = None  # Seconds until expiration
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def age_days(self) -> float:
        """Age in days."""
        return (time.time() - self.created_at) / 86400.0


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str    # "user" or "assistant" or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """An episodic memory — a summary of a past interaction."""

    episode_id: str
    title: str
    summary: str
    key_learnings: List[str] = field(default_factory=list)
    outcome: str = ""
    participants: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)

    def age_days(self) -> float:
        """Age in days."""
        return (time.time() - self.timestamp) / 86400.0


# ──────────────────────────────────────────────────────────────────
# Agent Memory
# ──────────────────────────────────────────────────────────────────


class AgentMemory:
    """Comprehensive memory system for agents.

    Manages multiple memory types:
    - Short-term: Recent conversation turns
    - Long-term: Persistent key-value store
    - Episodic: Summaries of past interactions
    - Working: Scratchpad for current task
    """

    def __init__(
        self,
        short_term_limit: int = 50,         # Max conversation turns
        long_term_limit: int = 1000,         # Max long-term entries
        episodic_limit: int = 100,           # Max episodes
        summarization_trigger: int = 20,     # Turns before auto-summarize
        embed_fn: Optional[Callable] = None,
    ):
        """Initialize the agent memory.

        Args:
            short_term_limit: Max conversation turns in short-term.
            long_term_limit: Max long-term memory entries.
            episodic_limit: Max episodic memories.
            summarization_trigger: Auto-summarize after this many turns.
            embed_fn: Function for generating embeddings.
        """
        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit
        self.episodic_limit = episodic_limit
        self.summarization_trigger = summarization_trigger
        self.embed_fn = embed_fn

        # Storage
        self._conversation: List[ConversationTurn] = []
        self._long_term: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._episodes: List[Episode] = []
        self._working_memory: Dict[str, Any] = {}
        self._summaries: List[str] = []

        # Stats
        self._total_turns = 0
        self._total_tokens = 0

    # ── Short-term Memory (Conversation) ──────────────────────────

    def add_turn(
        self,
        role: str,
        content: str,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation turn.

        Args:
            role: Message role.
            content: Message content.
            token_count: Token count estimate.
            metadata: Optional metadata.
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        self._conversation.append(turn)
        self._total_turns += 1
        self._total_tokens += token_count

        # Trim if over limit
        while len(self._conversation) > self.short_term_limit:
            self._conversation.pop(0)

        # Auto-summarize
        if len(self._conversation) >= self.summarization_trigger:
            self._maybe_summarize()

    def get_recent_turns(self, n: int = 10) -> List[ConversationTurn]:
        """Get the most recent conversation turns.

        Args:
            n: Number of turns.

        Returns:
            List of recent ConversationTurns.
        """
        return self._conversation[-n:]

    def get_conversation_context(self, max_tokens: int = 4000) -> str:
        """Get formatted conversation context for LLM.

        Args:
            max_tokens: Token limit for context.

        Returns:
            Formatted conversation string.
        """
        turns = self._conversation
        parts = []
        token_count = 0

        # Go from most recent backward
        for turn in reversed(turns):
            line = f"{turn.role}: {turn.content}"
            est_tokens = len(line.split())
            if token_count + est_tokens > max_tokens:
                break
            parts.insert(0, line)
            token_count += est_tokens

        return "\n".join(parts)

    def clear_conversation(self) -> None:
        """Clear short-term conversation memory."""
        self._conversation.clear()

    # ── Long-term Memory ──────────────────────────────────────────

    def remember(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a long-term memory.

        Args:
            key: Memory key.
            value: Memory value.
            importance: Importance score (0-1).
            ttl: Time-to-live in seconds.
            tags: Tags for organization.
            metadata: Additional metadata.
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=MemoryType.LONG_TERM,
            importance=importance,
            ttl=ttl,
            tags=tags or [],
            metadata=metadata or {},
        )

        if self.embed_fn:
            try:
                entry.embedding = self.embed_fn(str(value))
            except Exception:
                pass

        self._long_term[key] = entry
        # Move to end (most recent)
        self._long_term.move_to_end(key)

        # Evict if over limit (by importance)
        while len(self._long_term) > self.long_term_limit:
            # Remove least important entry
            min_key = min(
                self._long_term.keys(),
                key=lambda k: self._long_term[k].importance,
            )
            del self._long_term[min_key]

    def recall(self, key: str) -> Optional[Any]:
        """Recall a long-term memory.

        Args:
            key: Memory key.

        Returns:
            Memory value or None.
        """
        entry = self._long_term.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._long_term[key]
            return None
        entry.touch()
        self._long_term.move_to_end(key)
        return entry.value

    def forget(self, key: str) -> bool:
        """Remove a long-term memory.

        Args:
            key: Memory key.

        Returns:
            True if removed.
        """
        if key in self._long_term:
            del self._long_term[key]
            return True
        return False

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[str, Any, float]]:
        """Search long-term memory by keyword or embedding similarity.

        Args:
            query: Search query.
            top_k: Max results.
            min_importance: Minimum importance filter.
            tags: Filter by tags.

        Returns:
            List of (key, value, score) tuples.
        """
        results = []

        for key, entry in self._long_term.items():
            if entry.is_expired():
                continue
            if entry.importance < min_importance:
                continue
            if tags and not set(tags).intersection(entry.tags):
                continue

            # Score by keyword match
            score = self._keyword_score(query, key, str(entry.value))
            results.append((key, entry.value, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, Any, float]]:
        """Search memory using embedding similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Max results.

        Returns:
            List of (key, value, similarity) tuples.
        """
        scored = []
        for key, entry in self._long_term.items():
            if entry.embedding is None or entry.is_expired():
                continue
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            scored.append((key, entry.value, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def get_memories_by_tag(self, tag: str) -> List[Tuple[str, Any]]:
        """Get all memories with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of (key, value) tuples.
        """
        return [
            (k, e.value)
            for k, e in self._long_term.items()
            if tag in e.tags and not e.is_expired()
        ]

    # ── Episodic Memory ───────────────────────────────────────────

    def add_episode(
        self,
        title: str,
        summary: str,
        key_learnings: Optional[List[str]] = None,
        outcome: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record an episodic memory.

        Args:
            title: Episode title.
            summary: Summary of the interaction.
            key_learnings: Key lessons learned.
            outcome: Outcome of the interaction.
            importance: Importance score.
            tags: Organizational tags.

        Returns:
            Episode ID.
        """
        import uuid
        episode = Episode(
            episode_id=str(uuid.uuid4())[:8],
            title=title,
            summary=summary,
            key_learnings=key_learnings or [],
            outcome=outcome,
            importance=importance,
            tags=tags or [],
        )
        self._episodes.append(episode)

        # Evict oldest/least important
        while len(self._episodes) > self.episodic_limit:
            min_ep = min(self._episodes, key=lambda e: (e.importance, e.timestamp))
            self._episodes.remove(min_ep)

        return episode.episode_id

    def get_relevant_episodes(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Episode]:
        """Get episodes relevant to a query.

        Args:
            query: Search query.
            top_k: Max results.

        Returns:
            List of relevant Episodes.
        """
        scored = []
        for ep in self._episodes:
            score = self._keyword_score(query, ep.title, ep.summary)
            # Boost by importance
            score *= (1 + ep.importance)
            # Boost recent episodes
            recency = max(0, 1 - ep.age_days() / 30)  # Decay over 30 days
            score *= (1 + recency)
            scored.append((ep, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_k]]

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID.

        Args:
            episode_id: Episode ID.

        Returns:
            Episode or None.
        """
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    # ── Working Memory ────────────────────────────────────────────

    def set_working(self, key: str, value: Any) -> None:
        """Set a working memory variable.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self._working_memory[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a working memory variable.

        Args:
            key: Variable name.
            default: Default if not found.

        Returns:
            Variable value.
        """
        return self._working_memory.get(key, default)

    def clear_working(self) -> None:
        """Clear working memory."""
        self._working_memory.clear()

    # ── Summarization ─────────────────────────────────────────────

    def _maybe_summarize(self) -> None:
        """Automatically summarize conversation if needed."""
        # Mark that summarization should happen
        # Actual summarization requires LLM, done externally
        recent = self._conversation[-self.summarization_trigger:]
        # Simple extractive summary for now (first and last turns)
        if len(recent) >= 3:
            summary_parts = [
                f"User started discussing: {recent[0].content[:100]}...",
                f"Latest turn: {recent[-1].role}: {recent[-1].content[:100]}...",
            ]
            self._summaries.append(" | ".join(summary_parts))

    def get_summary(self) -> str:
        """Get conversation summary.

        Returns:
            Summary string.
        """
        if not self._conversation:
            return "No conversation history."

        turns = len(self._conversation)
        first = self._conversation[0]
        last = self._conversation[-1]

        return (
            f"Conversation with {turns} turns. "
            f"Started with: {first.content[:100]}... "
            f"Last message: {last.content[:100]}..."
        )

    # ── Utilities ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dict with memory stats.
        """
        now = time.time()
        return {
            "short_term_turns": len(self._conversation),
            "short_term_tokens": sum(t.token_count for t in self._conversation),
            "long_term_entries": len(self._long_term),
            "episodes": len(self._episodes),
            "working_memory_vars": len(self._working_memory),
            "summaries": len(self._summaries),
            "total_turns": self._total_turns,
            "total_tokens": self._total_tokens,
            "avg_importance": (
                sum(e.importance for e in self._long_term.values()) / max(len(self._long_term), 1)
            ),
            "expired_entries": sum(
                1 for e in self._long_term.values() if e.is_expired()
            ),
        }

    def clean_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired = [k for k, e in self._long_term.items() if e.is_expired()]
        for k in expired:
            del self._long_term[k]
        return len(expired)

    def clear_all(self) -> None:
        """Clear all memory."""
        self._conversation.clear()
        self._long_term.clear()
        self._episodes.clear()
        self._working_memory.clear()
        self._summaries.clear()
        self._total_turns = 0
        self._total_tokens = 0

    def export(self) -> Dict[str, Any]:
        """Export all memory as a serializable dict.

        Returns:
            Dict with all memories.
        """
        return {
            "conversation": [
                {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                for t in self._conversation
            ],
            "long_term": {
                k: {"value": str(e.value), "importance": e.importance, "tags": e.tags}
                for k, e in self._long_term.items()
            },
            "episodes": [
                {
                    "id": e.episode_id,
                    "title": e.title,
                    "summary": e.summary,
                    "learnings": e.key_learnings,
                    "importance": e.importance,
                }
                for e in self._episodes
            ],
            "working_memory": dict(self._working_memory),
            "stats": self.get_stats(),
        }

    def import_memory(self, data: Dict[str, Any]) -> None:
        """Import memory from exported dict.

        Args:
            data: Exported memory dict.
        """
        for turn_data in data.get("conversation", []):
            self.add_turn(
                role=turn_data["role"],
                content=turn_data["content"],
            )
        for key, entry_data in data.get("long_term", {}).items():
            self.remember(
                key=key,
                value=entry_data["value"],
                importance=entry_data.get("importance", 0.5),
                tags=entry_data.get("tags", []),
            )
        for ep_data in data.get("episodes", []):
            self.add_episode(
                title=ep_data["title"],
                summary=ep_data["summary"],
                key_learnings=ep_data.get("learnings", []),
                importance=ep_data.get("importance", 0.5),
            )
        for key, value in data.get("working_memory", {}).items():
            self.set_working(key, value)

    @staticmethod
    def _keyword_score(query: str, *texts: str) -> float:
        """Simple keyword overlap score.

        Args:
            query: Search query.
            *texts: Texts to score against.

        Returns:
            Score (0+).
        """
        query_terms = set(query.lower().split())
        combined = " ".join(texts).lower()
        text_terms = set(combined.split())
        overlap = query_terms & text_terms
        return len(overlap) / max(len(query_terms), 1)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = np.sqrt(sum(x * x for x in a))
        norm_b = np.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
