"""Cross-encoder reranker using BAAI/bge-reranker-base.

Scores candidate chunks against the query, applies sigmoid normalization
to the raw logits, and enforces a minimum score threshold before returning
the top-k results. Below threshold → signals "not found in document".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.config import settings
from app.exceptions import RerankerError

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", settings.reranker_model)
            _reranker = CrossEncoder(settings.reranker_model)
            logger.info("Reranker model loaded.")
        except Exception as exc:
            raise RerankerError(f"Failed to load reranker model: {exc}") from exc
    return _reranker


def _sigmoid(x: float) -> float:
    """Map raw logit to [0, 1] probability."""
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RankedChunk:
    """A chunk with a normalised relevance score."""
    chunk_id: str
    document_id: str
    content: str
    page_start: int
    page_end: int
    section_title: str
    token_count: int
    score: float  # sigmoid-normalized, 0-1

    @property
    def page_citation(self) -> str:
        if self.page_start == self.page_end:
            return f"p{self.page_start}"
        return f"p{self.page_start}-{self.page_end}"


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[RankedChunk]:
    """Rerank candidate chunks against the query using a cross-encoder.

    Args:
        query:      The user's question.
        candidates: List of chunk dicts from the hybrid retriever.
        top_k:      Number of top chunks to return (default from settings).
        min_score:  Minimum sigmoid score to include (default from settings).
                    If no chunk meets this threshold, returns an empty list —
                    the caller must handle this as "not found in document".

    Returns:
        Sorted list of RankedChunk objects (best first).

    Raises:
        RerankerError on model or scoring failure.
    """
    k = top_k or settings.rerank_top_k
    threshold = min_score if min_score is not None else settings.rerank_min_score

    if not candidates:
        logger.info("Reranker received 0 candidates — returning empty.")
        return []

    model = _get_reranker()

    try:
        pairs = [(query, c["content"]) for c in candidates]
        raw_scores = model.predict(pairs)
    except Exception as exc:
        raise RerankerError(f"Cross-encoder prediction failed: {exc}") from exc

    ranked: list[RankedChunk] = []
    for chunk, raw_score in zip(candidates, raw_scores):
        norm_score = _sigmoid(float(raw_score))
        ranked.append(RankedChunk(
            chunk_id=chunk.get("chunk_id", ""),
            document_id=chunk.get("document_id", ""),
            content=chunk.get("content", ""),
            page_start=chunk.get("page_start", 1),
            page_end=chunk.get("page_end", 1),
            section_title=chunk.get("section_title", ""),
            token_count=chunk.get("token_count", 0),
            score=norm_score,
        ))

    ranked.sort(key=lambda c: c.score, reverse=True)

    # Apply threshold — below min_score signals "no relevant content found"
    above_threshold = [r for r in ranked[:k] if r.score >= threshold]

    logger.info(
        "Reranker: %d candidates → top %d above threshold %.2f (best score: %.3f).",
        len(candidates),
        len(above_threshold),
        threshold,
        ranked[0].score if ranked else 0.0,
    )

    return above_threshold
