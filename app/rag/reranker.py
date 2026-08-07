"""Pass-through reranker to save RAM on Render.

Bypasses the heavy local cross-encoder model. 
Simply returns the top-k candidates exactly as they came from the vector store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)


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
    score: float  # Dummy score

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
    """Pass-through reranker. Returns the candidates as RankedChunks without ML reranking."""
    k = top_k or settings.rerank_top_k

    if not candidates:
        logger.info("Reranker received 0 candidates — returning empty.")
        return []

    ranked: list[RankedChunk] = []
    
    # Assign dummy descending scores based on original retrieval order
    for idx, chunk in enumerate(candidates):
        ranked.append(RankedChunk(
            chunk_id=chunk.get("chunk_id", ""),
            document_id=chunk.get("document_id", ""),
            content=chunk.get("content", ""),
            page_start=chunk.get("page_start", 1),
            page_end=chunk.get("page_end", 1),
            section_title=chunk.get("section_title", ""),
            token_count=chunk.get("token_count", 0),
            score=1.0 - (idx * 0.01),  # Dummy descending score
        ))

    top_chunks = ranked[:k]
    
    logger.info(
        "Dummy Reranker bypassed ML model. Returning top %d candidates.",
        len(top_chunks)
    )

    return top_chunks
