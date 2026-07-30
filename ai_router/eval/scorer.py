"""
Evaluation scoring for LLM outputs.

Provides multiple scoring metrics:
  - BLEU (machine translation quality)
  - ROUGE (summarization quality)
  - Semantic similarity (embedding-based)
  - Exact match
  - Token-level F1
  - Custom scoring functions
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ScorerType(str, Enum):
    """Available scorer types."""

    BLEU = "bleu"
    ROUGE = "rouge"
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    SEMANTIC = "semantic"
    EXACT_MATCH = "exact_match"
    F1 = "f1"
    CUSTOM = "custom"


@dataclass
class ScoreResult:
    """Result of a scoring operation."""

    score: float
    scorer_type: ScorerType
    details: Dict[str, Any] = field(default_factory=dict)
    reference: str = ""
    candidate: str = ""


@dataclass
class MultiScoreResult:
    """Multiple scores for a single evaluation pair."""

    reference: str
    candidate: str
    scores: Dict[str, ScoreResult] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        """Average across all scores."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)


# ──────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────


def tokenize(text: str, lang: str = "en") -> List[str]:
    """Tokenize text into words.

    Args:
        text: Input text.
        lang: Language code (en, zh, etc.).

    Returns:
        List of tokens.
    """
    if lang == "zh":
        # Simple Chinese tokenization (character-level)
        return list(text.replace(" ", ""))
    else:
        # English tokenization
        return text.lower().split()


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from tokens.

    Args:
        tokens: Token list.
        n: N-gram size.

    Returns:
        List of n-gram tuples.
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ──────────────────────────────────────────────────────────────────
# BLEU Score
# ──────────────────────────────────────────────────────────────────


class BLEUScorer:
    """BLEU (Bilingual Evaluation Understudy) scorer.

    Computes BLEU score with n-gram precision and brevity penalty.
    Standard for machine translation evaluation.
    """

    def __init__(self, max_n: int = 4, smooth: bool = True):
        """Initialize BLEU scorer.

        Args:
            max_n: Maximum n-gram order (default 4 for BLEU-4).
            smooth: Apply smoothing to avoid zero scores.
        """
        self.max_n = max_n
        self.smooth = smooth

    def score(
        self,
        candidate: str,
        reference: Union[str, List[str]],
        lang: str = "en",
    ) -> ScoreResult:
        """Compute BLEU score.

        Args:
            candidate: Candidate text.
            reference: Reference text or list of references.
            lang: Language code.

        Returns:
            ScoreResult with BLEU score.
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        cand_tokens = tokenize(candidate, lang)
        ref_tokens_list = [tokenize(ref, lang) for ref in references]

        if not cand_tokens:
            return ScoreResult(
                score=0.0,
                scorer_type=ScorerType.BLEU,
                details={"reason": "Empty candidate"},
            )

        # Compute precisions for each n
        precisions = []
        for n in range(1, self.max_n + 1):
            cand_ngrams = Counter(ngrams(cand_tokens, n))

            # Max reference count for each n-gram
            max_ref_counts: Dict[Tuple, int] = {}
            for ref_tokens in ref_tokens_list:
                ref_ngrams = Counter(ngrams(ref_tokens, n))
                for ng, count in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), count)

            # Clip counts
            clipped_count = 0
            total_count = 0
            for ng, count in cand_ngrams.items():
                total_count += count
                clipped_count += min(count, max_ref_counts.get(ng, 0))

            if self.smooth and total_count == 0:
                precisions.append(1.0)
            elif total_count == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped_count / total_count)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_sum = sum(math.log(p) for p in precisions)
            bleu = math.exp(log_sum / self.max_n)

        # Brevity penalty
        cand_len = len(cand_tokens)
        ref_len = min(len(rt) for rt in ref_tokens_list)
        if cand_len == 0:
            bp = 0.0
        elif cand_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / cand_len)

        final_bleu = bp * bleu

        return ScoreResult(
            score=final_bleu,
            scorer_type=ScorerType.BLEU,
            details={
                "precisions": precisions,
                "brevity_penalty": bp,
                "candidate_length": cand_len,
                "reference_length": ref_len,
            },
            reference=references[0],
            candidate=candidate,
        )


# ──────────────────────────────────────────────────────────────────
# ROUGE Score
# ──────────────────────────────────────────────────────────────────


class ROUGEScorer:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scorer.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.
    Standard for summarization evaluation.
    """

    def __init__(self, beta: float = 1.0):
        """Initialize ROUGE scorer.

        Args:
            beta: F-beta parameter (1.0 = F1).
        """
        self.beta = beta

    def score(
        self,
        candidate: str,
        reference: Union[str, List[str]],
        rouge_type: str = "rougeL",
        lang: str = "en",
    ) -> ScoreResult:
        """Compute ROUGE score.

        Args:
            candidate: Candidate text.
            reference: Reference text.
            rouge_type: 'rouge1', 'rouge2', or 'rougeL'.
            lang: Language code.

        Returns:
            ScoreResult with ROUGE scores.
        """
        if isinstance(reference, str):
            reference = reference

        cand_tokens = tokenize(candidate, lang)
        ref_tokens = tokenize(reference, lang)

        if not cand_tokens or not ref_tokens:
            return ScoreResult(
                score=0.0,
                scorer_type=ScorerType.ROUGE,
                details={"reason": "Empty input"},
            )

        if rouge_type == "rouge1":
            precision, recall, f1 = self._rouge_n(cand_tokens, ref_tokens, n=1)
            scorer_type = ScorerType.ROUGE1
        elif rouge_type == "rouge2":
            precision, recall, f1 = self._rouge_n(cand_tokens, ref_tokens, n=2)
            scorer_type = ScorerType.ROUGE2
        else:  # rougeL
            precision, recall, f1 = self._rouge_l(cand_tokens, ref_tokens)
            scorer_type = ScorerType.ROUGEL

        return ScoreResult(
            score=f1,
            scorer_type=scorer_type,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "beta": self.beta,
            },
            reference=reference,
            candidate=candidate,
        )

    def score_all(
        self,
        candidate: str,
        reference: str,
        lang: str = "en",
    ) -> MultiScoreResult:
        """Compute all ROUGE variants.

        Args:
            candidate: Candidate text.
            reference: Reference text.
            lang: Language code.

        Returns:
            MultiScoreResult with all ROUGE scores.
        """
        result = MultiScoreResult(reference=reference, candidate=candidate)

        for rtype in ["rouge1", "rouge2", "rougeL"]:
            result.scores[rtype] = self.score(candidate, reference, rtype, lang)

        return result

    def _rouge_n(
        self,
        cand_tokens: List[str],
        ref_tokens: List[str],
        n: int,
    ) -> Tuple[float, float, float]:
        """Compute ROUGE-N precision, recall, F1.

        Args:
            cand_tokens: Candidate tokens.
            ref_tokens: Reference tokens.
            n: N-gram size.

        Returns:
            Tuple of (precision, recall, f1).
        """
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        ref_ngrams = Counter(ngrams(ref_tokens, n))

        overlap = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams if ng in ref_ngrams)

        precision = overlap / max(len(cand_ngrams), 1)
        recall = overlap / max(len(ref_ngrams), 1)
        f1 = self._f_beta(precision, recall)

        return precision, recall, f1

    def _rouge_l(
        self,
        cand_tokens: List[str],
        ref_tokens: List[str],
    ) -> Tuple[float, float, float]:
        """Compute ROUGE-L (Longest Common Subsequence).

        Args:
            cand_tokens: Candidate tokens.
            ref_tokens: Reference tokens.

        Returns:
            Tuple of (precision, recall, f1).
        """
        lcs_len = self._lcs_length(cand_tokens, ref_tokens)

        precision = lcs_len / max(len(cand_tokens), 1)
        recall = lcs_len / max(len(ref_tokens), 1)
        f1 = self._f_beta(precision, recall)

        return precision, recall, f1

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """Compute length of Longest Common Subsequence.

        Args:
            a: First sequence.
            b: Second sequence.

        Returns:
            LCS length.
        """
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _f_beta(self, precision: float, recall: float) -> float:
        """Compute F-beta score.

        Args:
            precision: Precision value.
            recall: Recall value.

        Returns:
            F-beta score.
        """
        if precision + recall == 0:
            return 0.0
        beta_sq = self.beta * self.beta
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


# ──────────────────────────────────────────────────────────────────
# Semantic Similarity Scorer
# ──────────────────────────────────────────────────────────────────


class SemanticScorer:
    """Semantic similarity scorer using embeddings.

    Computes cosine similarity between candidate and reference embeddings.
    More robust than n-gram metrics for meaning preservation.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        threshold: float = 0.7,
    ):
        """Initialize semantic scorer.

        Args:
            embed_fn: Function to generate embeddings.
            threshold: Threshold for pass/fail classification.
        """
        self.embed_fn = embed_fn or self._default_embed_fn
        self.threshold = threshold

    @staticmethod
    def _default_embed_fn(text: str) -> List[float]:
        """Default embedding using simple word overlap vector."""
        import hashlib
        words = text.lower().split()
        vec = [0.0] * 128
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % 128
            vec[h] += 1.0
        norm = np.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def score(
        self,
        candidate: str,
        reference: str,
    ) -> ScoreResult:
        """Compute semantic similarity score.

        Args:
            candidate: Candidate text.
            reference: Reference text.

        Returns:
            ScoreResult with similarity score.
        """
        cand_emb = self.embed_fn(candidate)
        ref_emb = self.embed_fn(reference)

        dot = sum(a * b for a, b in zip(cand_emb, ref_emb))
        norm_a = np.sqrt(sum(a * a for a in cand_emb))
        norm_b = np.sqrt(sum(b * b for b in ref_emb))

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot / (norm_a * norm_b)

        return ScoreResult(
            score=similarity,
            scorer_type=ScorerType.SEMANTIC,
            details={
                "passes_threshold": similarity >= self.threshold,
                "threshold": self.threshold,
            },
            reference=reference,
            candidate=candidate,
        )


# ──────────────────────────────────────────────────────────────────
# Exact Match & F1 Scorer
# ──────────────────────────────────────────────────────────────────


class ExactMatchScorer:
    """Exact match scorer with normalization options."""

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = False,
    ):
        """Initialize exact match scorer.

        Args:
            ignore_case: Case-insensitive comparison.
            ignore_whitespace: Normalize whitespace.
            ignore_punctuation: Strip punctuation.
        """
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation

    def score(self, candidate: str, reference: str) -> ScoreResult:
        """Compute exact match score.

        Args:
            candidate: Candidate text.
            reference: Reference text.

        Returns:
            ScoreResult (1.0 or 0.0).
        """
        cand = self._normalize(candidate)
        ref = self._normalize(reference)

        return ScoreResult(
            score=1.0 if cand == ref else 0.0,
            scorer_type=ScorerType.EXACT_MATCH,
            details={"normalized_candidate": cand[:100], "normalized_reference": ref[:100]},
            reference=reference,
            candidate=candidate,
        )

    def _normalize(self, text: str) -> str:
        """Normalize text.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        if self.ignore_case:
            text = text.lower()
        if self.ignore_whitespace:
            text = " ".join(text.split())
        if self.ignore_punctuation:
            import re
            text = re.sub(r'[^\w\s]', '', text)
        return text


class TokenF1Scorer:
    """Token-level F1 scorer."""

    def score(
        self,
        candidate: str,
        reference: str,
        lang: str = "en",
    ) -> ScoreResult:
        """Compute token-level F1.

        Args:
            candidate: Candidate text.
            reference: Reference text.
            lang: Language code.

        Returns:
            ScoreResult with F1 score.
        """
        cand_tokens = set(tokenize(candidate, lang))
        ref_tokens = set(tokenize(reference, lang))

        if not cand_tokens or not ref_tokens:
            return ScoreResult(
                score=0.0,
                scorer_type=ScorerType.F1,
                details={"reason": "Empty tokens"},
            )

        overlap = cand_tokens & ref_tokens

        precision = len(overlap) / len(cand_tokens)
        recall = len(overlap) / len(ref_tokens)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return ScoreResult(
            score=f1,
            scorer_type=ScorerType.F1,
            details={
                "precision": precision,
                "recall": recall,
                "overlap_count": len(overlap),
            },
            reference=reference,
            candidate=candidate,
        )


# ──────────────────────────────────────────────────────────────────
# Unified Scorer
# ──────────────────────────────────────────────────────────────────


class Scorer:
    """Unified scoring interface combining all metrics."""

    def __init__(self, embed_fn: Optional[Callable] = None):
        """Initialize the scorer.

        Args:
            embed_fn: Embedding function for semantic scoring.
        """
        self.bleu = BLEUScorer()
        self.rouge = ROUGEScorer()
        self.semantic = SemanticScorer(embed_fn=embed_fn)
        self.exact_match = ExactMatchScorer()
        self.token_f1 = TokenF1Scorer()

    def score(
        self,
        candidate: str,
        reference: str,
        metrics: Optional[List[ScorerType]] = None,
        lang: str = "en",
    ) -> MultiScoreResult:
        """Score a candidate against a reference.

        Args:
            candidate: Candidate text.
            reference: Reference text.
            metrics: List of metrics to compute (all if None).
            lang: Language code.

        Returns:
            MultiScoreResult with all requested scores.
        """
        metrics = metrics or list(ScorerType)

        result = MultiScoreResult(reference=reference, candidate=candidate)

        for metric in metrics:
            if metric == ScorerType.BLEU:
                result.scores["bleu"] = self.bleu.score(candidate, reference, lang)
            elif metric == ScorerType.ROUGE1:
                result.scores["rouge1"] = self.rouge.score(candidate, reference, "rouge1", lang)
            elif metric == ScorerType.ROUGE2:
                result.scores["rouge2"] = self.rouge.score(candidate, reference, "rouge2", lang)
            elif metric == ScorerType.ROUGEL:
                result.scores["rougeL"] = self.rouge.score(candidate, reference, "rougeL", lang)
            elif metric == ScorerType.ROUGE:
                rouge_all = self.rouge.score_all(candidate, reference, lang)
                result.scores.update(rouge_all.scores)
            elif metric == ScorerType.SEMANTIC:
                result.scores["semantic"] = self.semantic.score(candidate, reference)
            elif metric == ScorerType.EXACT_MATCH:
                result.scores["exact_match"] = self.exact_match.score(candidate, reference)
            elif metric == ScorerType.F1:
                result.scores["f1"] = self.token_f1.score(candidate, reference, lang)

        return result

    def score_batch(
        self,
        candidates: List[str],
        references: List[str],
        metrics: Optional[List[ScorerType]] = None,
        lang: str = "en",
    ) -> List[MultiScoreResult]:
        """Score multiple candidate-reference pairs.

        Args:
            candidates: Candidate texts.
            references: Reference texts.
            metrics: Metrics to compute.
            lang: Language code.

        Returns:
            List of MultiScoreResults.
        """
        return [
            self.score(c, r, metrics, lang)
            for c, r in zip(candidates, references)
        ]

    def aggregate_scores(
        self,
        results: List[MultiScoreResult],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate scores across multiple results.

        Args:
            results: List of MultiScoreResults.

        Returns:
            Dict with metric -> {mean, min, max, std} stats.
        """
        if not results:
            return {}

        metric_scores: Dict[str, List[float]] = defaultdict(list)
        for res in results:
            for name, sr in res.scores.items():
                metric_scores[name].append(sr.score)

        aggregated = {}
        for name, scores in metric_scores.items():
            aggregated[name] = {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "count": len(scores),
            }

        return aggregated
