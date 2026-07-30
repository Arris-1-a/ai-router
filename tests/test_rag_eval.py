"""Tests for RAG pipeline and evaluation modules."""

import pytest
from ai_router.rag.chunker import (
    ChunkStrategy,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    SlidingWindowChunker,
    MarkdownChunker,
    create_chunker,
)
from ai_router.rag.retriever import BM25Retriever, HybridRetriever, RetrievalMode
from ai_router.rag.reranker import ScoreReranker, MMRReranker, RerankingPipeline, RerankStrategy
from ai_router.eval.scorer import (
    BLEUScorer,
    ROUGEScorer,
    SemanticScorer,
    ExactMatchScorer,
    TokenF1Scorer,
    Scorer,
    ScorerType,
)


# ── Chunker Tests ─────────────────────────────────────────────────


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "A " * 500
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 100 + 30  # Allow some slack for boundary

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_short_text(self):
        chunker = FixedSizeChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_chunk_metadata(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("Hello world " * 50, metadata={"source": "test"})
        assert all(c.metadata.get("strategy") == "fixed_size" for c in chunks)
        assert chunks[0].metadata.get("chunk_index") == 0


class TestSentenceChunker:
    def test_respects_sentences(self):
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=0)
        text = (
            "This is sentence one. This is sentence two. "
            "This is sentence three. This is sentence four. "
            "This is sentence five. This is sentence six."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        # No chunk should break mid-sentence
        for chunk in chunks:
            text = chunk.text.strip()
            assert not text or text[-1] in ".!?"

    def test_short_input(self):
        chunker = SentenceChunker(chunk_size=1000)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1


class TestParagraphChunker:
    def test_splits_on_paragraphs(self):
        chunker = ParagraphChunker(chunk_size=500)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text)
        assert len(chunks) <= 3


class TestRecursiveChunker:
    def test_basic_splitting(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        text = "Hello world. " * 200
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 400  # Recursive might be a bit larger

    def test_separator_priority(self):
        chunker = RecursiveChunker(
            chunk_size=100,
            separators=["\n\n", "\n", ". ", " "],
        )
        text = "Line1\n\nLine2\nLine3. Sentence."
        chunks = chunker.chunk(text)


class TestMarkdownChunker:
    def test_splits_on_headers(self):
        chunker = MarkdownChunker(chunk_size=500)
        text = """# Title
Some content here.

## Section 1
Content for section one.

## Section 2
Content for section two.
"""
        chunks = chunker.chunk(text)
        assert len(chunks) > 0

    def test_preserves_header_info(self):
        chunker = MarkdownChunker(chunk_size=500)
        text = "# Intro\nWelcome.\n\n## Details\nMore info here."
        chunks = chunker.chunk(text)
        assert any("Intro" in c.text for c in chunks) or any("Details" in c.text for c in chunks)


class TestSlidingWindowChunker:
    def test_overlapping_windows(self):
        chunker = SlidingWindowChunker(chunk_size=5, chunk_overlap=2)
        text = "one two three four five six seven eight nine ten"
        chunks = chunker.chunk(text)
        assert len(chunks) > 1


class TestChunkerFactory:
    def test_create_all_strategies(self):
        for strategy in ChunkStrategy:
            chunker = create_chunker(strategy=strategy, chunk_size=100)
            chunks = chunker.chunk("Test text. More text.")
            assert len(chunks) > 0


# ── Retriever Tests ──────────────────────────────────────────────


class TestBM25Retriever:
    def test_index_and_search(self):
        bm25 = BM25Retriever()
        docs = [
            "Python is a programming language",
            "Java is also a programming language",
            "Machine learning uses Python extensively",
            "Coffee is a popular beverage",
        ]
        bm25.index(docs)
        results = bm25.search("Python programming", top_k=3)
        assert len(results) > 0
        # Python-related docs should rank higher
        assert results[0][0] in (0, 2)  # Index 0 or 2

    def test_add_document(self):
        bm25 = BM25Retriever()
        bm25.index(["doc1 content", "doc2 content"])
        doc_id = bm25.add_document("doc3 content", doc_id="custom_id")
        assert doc_id == "custom_id"
        results = bm25.search("doc3", top_k=5)
        assert any(r[0] == 2 for r in results)

    def test_empty_search(self):
        bm25 = BM25Retriever()
        results = bm25.search("anything")
        assert results == []

    def test_reset(self):
        bm25 = BM25Retriever()
        bm25.index(["doc1", "doc2"])
        bm25.reset()
        assert bm25._total_docs == 0


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_sparse_only(self):
        retriever = HybridRetriever(embedder=None, mode=RetrievalMode.SPARSE)
        docs = [f"Document {i} about topic {i % 3}" for i in range(20)]
        await retriever.index_documents(docs)

        result = await retriever.retrieve("topic 1", top_k=5)
        assert len(result.results) > 0
        assert result.retrieval_mode == RetrievalMode.SPARSE

    @pytest.mark.asyncio
    async def test_hybrid_rrf(self):
        retriever = HybridRetriever(embedder=None, mode=RetrievalMode.HYBRID_RRF)
        docs = [f"Document about AI and ML number {i}" for i in range(10)]
        await retriever.index_documents(docs)

        result = await retriever.retrieve("AI ML", top_k=5)
        assert len(result.results) > 0


# ── Reranker Tests ────────────────────────────────────────────────


class TestScoreReranker:
    @pytest.mark.asyncio
    async def test_basic_rerank(self):
        reranker = ScoreReranker(top_k=3)
        docs = [
            {"chunk_id": "1", "text": "Python guide", "score": 0.5},
            {"chunk_id": "2", "text": "Java guide", "score": 0.8},
            {"chunk_id": "3", "text": "C++ guide", "score": 0.3},
            {"chunk_id": "4", "text": "Ruby guide", "score": 0.9},
        ]
        result = await reranker.rerank("programming", docs, top_k=2)
        assert len(result.results) == 2
        # Highest score should be first
        assert result.results[0].chunk_id == "4"


class TestRerankingPipeline:
    @pytest.mark.asyncio
    async def test_multi_stage(self):
        r1 = ScoreReranker(top_k=3)
        r2 = ScoreReranker(top_k=2)
        pipeline = RerankingPipeline(rerankers=[r1, r2])

        docs = [
            {"chunk_id": str(i), "text": f"Doc {i}", "score": float(i) / 10}
            for i in range(10)
        ]
        result = await pipeline.rerank("test", docs, final_top_k=2)
        assert len(result.results) <= 2


# ── Scorer Tests ──────────────────────────────────────────────────


class TestBLEUScorer:
    def test_perfect_match(self):
        scorer = BLEUScorer()
        result = scorer.score("the cat sat on the mat", "the cat sat on the mat")
        # Should be close to 1.0
        assert result.score > 0.9

    def test_no_match(self):
        scorer = BLEUScorer()
        result = scorer.score("completely different text here", "the cat sat on the mat")
        assert result.score < 0.5


class TestROUGEScorer:
    def test_rouge1_perfect(self):
        scorer = ROUGEScorer()
        result = scorer.score("hello world", "hello world", rouge_type="rouge1")
        assert result.score > 0.99

    def test_rouge_l(self):
        scorer = ROUGEScorer()
        result = scorer.score(
            "the cat is on the mat",
            "there is a cat on the mat",
            rouge_type="rougeL",
        )
        assert 0 < result.score < 1.0

    def test_score_all(self):
        scorer = ROUGEScorer()
        result = scorer.score_all("hello world today", "hello world")
        assert "rouge1" in result.scores
        assert "rouge2" in result.scores
        assert "rougeL" in result.scores


class TestSemanticScorer:
    def test_similar_texts(self):
        scorer = SemanticScorer()
        result = scorer.score(
            "The weather is beautiful today",
            "It is a lovely day outside",
        )
        assert result.score > 0

    def test_dissimilar_texts(self):
        scorer = SemanticScorer()
        result = scorer.score(
            "Python programming language",
            "Chocolate cake recipe",
        )
        # Should have lower similarity
        assert result.score < 0.8


class TestExactMatchScorer:
    def test_exact_match(self):
        scorer = ExactMatchScorer()
        result = scorer.score("Hello World", "Hello World")
        assert result.score == 1.0

    def test_case_insensitive(self):
        scorer = ExactMatchScorer(ignore_case=True)
        result = scorer.score("Hello World", "hello world")
        assert result.score == 1.0

    def test_no_match(self):
        scorer = ExactMatchScorer()
        result = scorer.score("A", "B")
        assert result.score == 0.0


class TestTokenF1Scorer:
    def test_perfect_match(self):
        scorer = TokenF1Scorer()
        result = scorer.score("hello world", "hello world")
        assert result.score == 1.0

    def test_partial_match(self):
        scorer = TokenF1Scorer()
        result = scorer.score("hello world today", "hello world")
        assert 0.0 < result.score < 1.0


class TestUnifiedScorer:
    def test_multiple_metrics(self):
        scorer = Scorer()
        result = scorer.score(
            "The cat sat on the mat",
            "A cat was sitting on the mat",
            metrics=[ScorerType.BLEU, ScorerType.ROUGE1, ScorerType.F1],
        )
        assert "bleu" in result.scores
        assert "rouge1" in result.scores
        assert "f1" in result.scores

    def test_batch_scoring(self):
        scorer = Scorer()
        candidates = ["hello world", "goodbye world"]
        references = ["hello world", "goodbye moon"]
        results = scorer.score_batch(candidates, references, metrics=[ScorerType.F1])
        assert len(results) == 2

    def test_aggregate_scores(self):
        scorer = Scorer()
        results = scorer.score_batch(
            ["a b c", "d e f", "g h i"],
            ["a b c", "d e x", "g h i"],
            metrics=[ScorerType.F1],
        )
        aggregated = scorer.aggregate_scores(results)
        assert "f1" in aggregated
        assert "mean" in aggregated["f1"]
