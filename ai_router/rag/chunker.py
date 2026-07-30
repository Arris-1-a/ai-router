"""
Multi-strategy text chunker for RAG pipelines.

Provides various chunking strategies optimized for different document types:
  - Fixed-size with overlap
  - Sentence-aware chunking
  - Paragraph-based chunking
  - Recursive character splitting
  - Semantic/section-based chunking
  - Markdown-aware chunking
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class TextChunk:
    """A single text chunk with metadata."""

    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


# ──────────────────────────────────────────────────────────────────
# Base Chunker
# ──────────────────────────────────────────────────────────────────


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: Optional[Callable[[str], int]] = None,
        separators: Optional[List[str]] = None,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size (in tokens or chars).
            chunk_overlap: Overlap between chunks.
            tokenizer: Function to count tokens (default: char-based).
            separators: Priority-ordered list of separators.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or (lambda x: len(x))
        self.separators = separators or ["\n\n", "\n", ". ", "。", " ", ""]

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text into chunks.

        Args:
            text: Input text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of TextChunk objects.
        """
        ...

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.tokenizer(text)

    def _create_chunks(
        self,
        segments: List[str],
        parent_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Create TextChunk objects from text segments.

        Args:
            segments: List of text segments.
            parent_metadata: Metadata to merge into each chunk.

        Returns:
            List of TextChunk objects.
        """
        chunks = []
        char_pos = 0
        for i, segment in enumerate(segments):
            end_pos = char_pos + len(segment)
            chunks.append(TextChunk(
                text=segment,
                index=i,
                start_char=char_pos,
                end_char=end_pos,
                token_count=self._count_tokens(segment),
                metadata={**(parent_metadata or {}), "chunk_index": i},
            ))
            char_pos = end_pos
        return chunks


# ──────────────────────────────────────────────────────────────────
# Fixed-Size Chunker
# ──────────────────────────────────────────────────────────────────


class FixedSizeChunker(Chunker):
    """Simple fixed-size chunker with character/token-based splitting.

    Splits text into chunks of approximately `chunk_size` tokens/characters
    with `chunk_overlap` overlap between consecutive chunks.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text into fixed-size chunks.

        Args:
            text: Input text.
            metadata: Optional metadata.

        Returns:
            List of fixed-size TextChunks.
        """
        if not text.strip():
            return []

        chunks: List[TextChunk] = []
        start = 0
        text_len = len(text)
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to break at a natural boundary
            if end < text_len:
                # Look for the last sentence break within the chunk
                for sep in [". ", "\n", " "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    token_count=self._count_tokens(chunk_text),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": idx,
                        "strategy": "fixed_size",
                    },
                ))
                idx += 1

            start = end - self.chunk_overlap
            if start >= text_len:
                break

        return chunks


# ──────────────────────────────────────────────────────────────────
# Sentence Chunker
# ──────────────────────────────────────────────────────────────────


class SentenceChunker(Chunker):
    """Sentence-aware chunker that respects sentence boundaries.

    Splits text by sentences and groups them into chunks that fit
    within the chunk_size, without breaking sentences in the middle.
    """

    # Regex for sentence boundaries (supports English and Chinese)
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?。！？\n])\s+',
    )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text into sentence-aware chunks.

        Args:
            text: Input text.
            metadata: Optional metadata.

        Returns:
            List of sentence-grouped TextChunks.
        """
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        current_chunk: List[str] = []
        current_size = 0
        char_pos = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_size = self._count_tokens(sentence)

            if current_size + sent_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=chunk_idx,
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                    token_count=self._count_tokens(chunk_text),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": chunk_idx,
                        "strategy": "sentence",
                    },
                ))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(self._count_tokens(s) for s in current_chunk)
                chunk_idx += 1

            current_chunk.append(sentence)
            current_size += sent_size
            char_pos += len(sentence)

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=chunk_idx,
                start_char=char_pos - len(chunk_text),
                end_char=char_pos,
                token_count=self._count_tokens(chunk_text),
                metadata={
                    **(metadata or {}),
                    "chunk_index": chunk_idx,
                    "strategy": "sentence",
                },
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Split on sentence boundaries
        parts = self.SENTENCE_PATTERN.split(text)
        # Rejoin short fragments
        sentences = []
        buffer = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(buffer) + len(part) < 20:
                buffer += " " + part
            else:
                if buffer:
                    sentences.append(buffer.strip())
                buffer = part
        if buffer:
            sentences.append(buffer.strip())
        return sentences

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap.

        Args:
            sentences: Current chunk sentences.

        Returns:
            Overlap sentences (last ~20%).
        """
        if not sentences:
            return []
        overlap_count = max(1, len(sentences) // 5)
        return sentences[-overlap_count:]


# ──────────────────────────────────────────────────────────────────
# Paragraph Chunker
# ──────────────────────────────────────────────────────────────────


class ParagraphChunker(Chunker):
    """Paragraph-based chunker that preserves document structure.

    Splits on paragraph boundaries and merges paragraphs into
    chunks that fit within the size limit.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text by paragraphs.

        Args:
            text: Input text.
            metadata: Optional metadata.

        Returns:
            List of paragraph-based TextChunks.
        """
        if not text.strip():
            return []

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: List[TextChunk] = []
        current_paras: List[str] = []
        current_size = 0
        char_pos = 0
        chunk_idx = 0

        for para in paragraphs:
            para_size = self._count_tokens(para)

            # If a single paragraph exceeds chunk_size, split it further
            if para_size > self.chunk_size:
                # Flush current chunk first
                if current_paras:
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        index=chunk_idx,
                        start_char=char_pos - len(chunk_text),
                        end_char=char_pos,
                        token_count=self._count_tokens(chunk_text),
                        metadata={
                            **(metadata or {}),
                            "chunk_index": chunk_idx,
                            "strategy": "paragraph",
                        },
                    ))
                    current_paras = []
                    current_size = 0
                    chunk_idx += 1

                # Split long paragraph into sentences
                sub_chunker = SentenceChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    tokenizer=self.tokenizer,
                )
                sub_chunks = sub_chunker.chunk(para, metadata={
                    **(metadata or {}),
                    "strategy": "paragraph_sentence_split",
                })
                for sc in sub_chunks:
                    sc.index = chunk_idx
                    sc.metadata["chunk_index"] = chunk_idx
                    chunks.append(sc)
                    chunk_idx += 1
                char_pos += len(para)
                continue

            if current_size + para_size > self.chunk_size and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=chunk_idx,
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                    token_count=self._count_tokens(chunk_text),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": chunk_idx,
                        "strategy": "paragraph",
                    },
                ))
                current_paras = []
                current_size = 0
                chunk_idx += 1

            current_paras.append(para)
            current_size += para_size
            char_pos += len(para)

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(TextChunk(
                text=chunk_text,
                index=chunk_idx,
                start_char=char_pos - len(chunk_text),
                end_char=char_pos,
                token_count=self._count_tokens(chunk_text),
                metadata={
                    **(metadata or {}),
                    "chunk_index": chunk_idx,
                    "strategy": "paragraph",
                },
            ))

        return chunks


# ──────────────────────────────────────────────────────────────────
# Recursive Chunker
# ──────────────────────────────────────────────────────────────────


class RecursiveChunker(Chunker):
    """Recursive character text splitter.

    Tries to split on increasingly granular separators,
    falling back to character-level splitting when necessary.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Recursively split text.

        Args:
            text: Input text.
            metadata: Optional metadata.

        Returns:
            List of TextChunks.
        """
        if not text.strip():
            return []

        split_texts = self._split_text(text)
        return self._create_chunks(split_texts, {
            **(metadata or {}),
            "strategy": "recursive",
        })

    def _split_text(self, text: str) -> List[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split.

        Returns:
            List of text segments.
        """
        if self._count_tokens(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for separator in self.separators:
            if separator == "":
                # Character-level split
                return self._char_split(text)

            splits = text.split(separator)
            if len(splits) > 1:
                # Merge small splits
                merged = self._merge_splits(splits, separator)
                if len(merged) > 1:
                    return merged

        return self._char_split(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits that are too small.

        Args:
            splits: List of text splits.
            separator: The separator used.

        Returns:
            Merged list.
        """
        result: List[str] = []
        current: List[str] = []
        current_size = 0

        for split in splits:
            size = self._count_tokens(split)
            if current_size + size > self.chunk_size and current:
                result.append(separator.join(current))
                # Handle overlap
                overlap_text = separator.join(current)
                overlap_tokens = self._count_tokens(overlap_text)
                while overlap_tokens > self.chunk_overlap and len(current) > 1:
                    current.pop(0)
                    overlap_text = separator.join(current)
                    overlap_tokens = self._count_tokens(overlap_text)
                current_size = sum(self._count_tokens(s) for s in current)

            if size > self.chunk_size:
                # Recursively split long splits
                if current:
                    result.append(separator.join(current))
                    current = []
                    current_size = 0
                sub = self._split_text(split)
                result.extend(sub)
            else:
                current.append(split)
                current_size += size

        if current:
            result.append(separator.join(current))

        return result

    def _char_split(self, text: str) -> List[str]:
        """Split text into chunks of chunk_size characters.

        Args:
            text: Text to split.

        Returns:
            Character-sized chunks.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


# ──────────────────────────────────────────────────────────────────
# Markdown Chunker
# ──────────────────────────────────────────────────────────────────


class MarkdownChunker(Chunker):
    """Markdown-aware chunker that respects headers and code blocks.

    Preserves markdown structure by splitting on headers first,
    then further subdividing large sections.
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split markdown text preserving structure.

        Args:
            text: Markdown text.
            metadata: Optional metadata.

        Returns:
            List of structure-aware TextChunks.
        """
        if not text.strip():
            return []

        sections = self._split_by_headers(text)
        chunks: List[TextChunk] = []
        char_pos = 0
        chunk_idx = 0

        for section_title, section_text in sections:
            sec_size = self._count_tokens(section_text)

            if sec_size <= self.chunk_size:
                display = f"{section_title}\n{section_text}" if section_title else section_text
                chunks.append(TextChunk(
                    text=display.strip(),
                    index=chunk_idx,
                    start_char=char_pos,
                    end_char=char_pos + len(display),
                    token_count=self._count_tokens(display),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": chunk_idx,
                        "strategy": "markdown",
                        "section": section_title,
                    },
                ))
                char_pos += len(display)
                chunk_idx += 1
            else:
                # Split large section using paragraph chunker
                sub = ParagraphChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    tokenizer=self.tokenizer,
                )
                sub_chunks = sub.chunk(section_text, metadata={
                    **(metadata or {}),
                    "strategy": "markdown_sub",
                    "section": section_title,
                })
                for sc in sub_chunks:
                    if section_title:
                        sc.text = f"{section_title}\n{sc.text}"
                    sc.index = chunk_idx
                    sc.metadata["chunk_index"] = chunk_idx
                    chunks.append(sc)
                    chunk_idx += 1
                char_pos += len(section_text)

        return chunks

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split markdown text by headers.

        Args:
            text: Markdown text.

        Returns:
            List of (header_title, content) tuples.
        """
        # Find all header positions
        headers = list(self.HEADER_PATTERN.finditer(text))

        if not headers:
            return [("", text)]

        sections = []
        for i, match in enumerate(headers):
            title = match.group(1)
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            content = text[start:end].strip()
            sections.append((title, content))

        # If text before first header, add it
        if headers[0].start() > 0:
            prefix = text[: headers[0].start()].strip()
            if prefix:
                sections.insert(0, ("", prefix))

        return sections


# ──────────────────────────────────────────────────────────────────
# Sliding Window Chunker
# ──────────────────────────────────────────────────────────────────


class SlidingWindowChunker(Chunker):
    """Sliding window chunker for maximum retrieval coverage.

    Creates overlapping chunks with a sliding window approach,
    ensuring every part of the text appears in multiple chunks.
    """

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Create sliding window chunks.

        Args:
            text: Input text.
            metadata: Optional metadata.

        Returns:
            List of overlapping TextChunks.
        """
        if not text.strip():
            return []

        tokens = self._tokenize_to_words(text)
        if not tokens:
            return []

        chunks: List[TextChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0

        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            chunk_text = " ".join(window_tokens)
            chunks.append(TextChunk(
                text=chunk_text,
                index=idx,
                start_char=start,
                end_char=end,
                token_count=len(window_tokens),
                metadata={
                    **(metadata or {}),
                    "chunk_index": idx,
                    "strategy": "sliding_window",
                    "window_start": start,
                    "window_end": end,
                },
            ))
            idx += 1
            if end >= len(tokens):
                break

        return chunks

    def _tokenize_to_words(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of word tokens.
        """
        return re.findall(r'\S+', text)


# ──────────────────────────────────────────────────────────────────
# Chunker Factory
# ──────────────────────────────────────────────────────────────────


_CHUNKER_REGISTRY: Dict[ChunkStrategy, type] = {
    ChunkStrategy.FIXED_SIZE: FixedSizeChunker,
    ChunkStrategy.SENTENCE: SentenceChunker,
    ChunkStrategy.PARAGRAPH: ParagraphChunker,
    ChunkStrategy.RECURSIVE: RecursiveChunker,
    ChunkStrategy.MARKDOWN: MarkdownChunker,
    ChunkStrategy.SLIDING_WINDOW: SlidingWindowChunker,
}


def create_chunker(
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    tokenizer: Optional[Callable[[str], int]] = None,
    **kwargs: Any,
) -> Chunker:
    """Create a chunker instance.

    Args:
        strategy: Chunking strategy to use.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        tokenizer: Token counting function.
        **kwargs: Strategy-specific arguments.

    Returns:
        A Chunker instance.

    Raises:
        ValueError: If strategy is unknown.
    """
    if strategy not in _CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunk strategy: {strategy}. "
            f"Available: {list(_CHUNKER_REGISTRY.keys())}"
        )
    cls = _CHUNKER_REGISTRY[strategy]
    return cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        **kwargs,
    )
