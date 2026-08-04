"""Hybrid retriever — merges and deduplicates results from Chroma and BM25.

Searches are always scoped to a single document_id to prevent
cross-document contamination.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.exceptions import RetrievalError
from app.rag import bm25_index, vector_store
from app.rag.embed import embed_query

logger = logging.getLogger(__name__)


def retrieve(
    query_text: str,
    document_id: str,
    top_k: int | None = None,
) -> list[dict]:
    """Run hybrid (vector + BM25) retrieval for a query, scoped to document_id.

    Returns a merged, deduplicated list of up to (2 × top_k) candidate chunks,
    each with normalised fields ready for the reranker.

    Raises RetrievalError on failure.
    """
    k = top_k or settings.retrieval_top_k

    if not query_text or not query_text.strip():
        raise RetrievalError("Cannot retrieve with an empty query.")

    logger.info("Hybrid retrieve: query='%s...' document_id=%s top_k=%d", query_text[:60], document_id, k)

    # --- Vector search ---
    try:
        query_vec = embed_query(query_text)
        vector_hits = vector_store.query_chunks(query_vec, document_id=document_id, top_k=k)
    except Exception as exc:
        raise RetrievalError(f"Vector retrieval failed: {exc}") from exc

    # --- BM25 keyword search ---
    try:
        bm25_hits = bm25_index.query(document_id, query_text, top_k=k)
    except Exception as exc:
        logger.warning("BM25 retrieval failed (non-fatal, continuing with vector only): %s", exc)
        bm25_hits = []

    # --- Merge and deduplicate by chunk_id ---
    seen: set[str] = set()
    merged: list[dict] = []

    for hit in vector_hits + bm25_hits:
        cid = hit.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            # Normalise field names across both sources
            merged.append({
                "chunk_id": cid,
                "document_id": hit.get("document_id", document_id),
                "content": hit.get("content", ""),
                "page_start": hit.get("page_start", 1),
                "page_end": hit.get("page_end", 1),
                "section_title": hit.get("section_title", ""),
                "token_count": hit.get("token_count", 0),
            })

    logger.info(
        "Hybrid merge: %d vector + %d BM25 → %d unique candidates.",
        len(vector_hits),
        len(bm25_hits),
        len(merged),
    )
    return merged
