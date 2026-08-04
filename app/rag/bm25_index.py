"""BM25 keyword search index — per-document, persisted to disk as a pickle.

On startup, rebuilds from ChromaDB metadata if no pickle is found.
Searches are always scoped to a single document_id.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import settings
from app.exceptions import RetrievalError

logger = logging.getLogger(__name__)

# In-memory registry: document_id → BM25Okapi instance
_indexes: dict[str, BM25Okapi] = {}
# Parallel registry: document_id → list of chunk dicts (for returning full metadata)
_chunk_registry: dict[str, list[dict]] = {}

_PICKLE_DIR = os.path.join(settings.chroma_persist_dir, "bm25")


def _pickle_path(document_id: str) -> str:
    return os.path.join(_PICKLE_DIR, f"{document_id}.pkl")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_index(document_id: str, chunks: list[dict]) -> None:
    """Build a BM25 index for a document and persist it to disk.

    Args:
        document_id: Stable document ID (used as the index key).
        chunks: List of dicts with at minimum 'content' key.

    Raises RetrievalError on failure.
    """
    if not chunks:
        raise RetrievalError(f"Cannot build BM25 index: no chunks for document_id={document_id}")

    os.makedirs(_PICKLE_DIR, exist_ok=True)

    try:
        tokenized = [_tokenize(c["content"]) for c in chunks]
        index = BM25Okapi(tokenized)

        _indexes[document_id] = index
        _chunk_registry[document_id] = chunks

        with open(_pickle_path(document_id), "wb") as f:
            pickle.dump({"index": index, "chunks": chunks}, f)

        logger.info("BM25 index built and persisted for document_id=%s (%d chunks).", document_id, len(chunks))
    except Exception as exc:
        raise RetrievalError(f"BM25 index build failed for document_id={document_id}: {exc}") from exc


def load_index(document_id: str) -> bool:
    """Load a previously pickled BM25 index into memory.

    Returns True if loaded successfully, False if no pickle exists.
    """
    path = _pickle_path(document_id)
    if not os.path.exists(path):
        return False

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        _indexes[document_id] = data["index"]
        _chunk_registry[document_id] = data["chunks"]
        logger.info("BM25 index loaded from disk for document_id=%s.", document_id)
        return True
    except Exception as exc:
        logger.warning("Failed to load BM25 pickle for %s: %s", document_id, exc)
        return False


def _ensure_index(document_id: str) -> None:
    """Ensure index is in memory, loading from disk if needed."""
    if document_id not in _indexes:
        loaded = load_index(document_id)
        if not loaded:
            # Rebuild from ChromaDB
            from app.rag.vector_store import get_all_chunks_for_document
            logger.info("No BM25 pickle found for %s — rebuilding from ChromaDB.", document_id)
            chunks = get_all_chunks_for_document(document_id)
            if not chunks:
                raise RetrievalError(f"No chunks found in ChromaDB to rebuild BM25 for document_id={document_id}")
            build_index(document_id, chunks)


def query(document_id: str, query_text: str, top_k: int | None = None) -> list[dict]:
    """Search BM25 index for a document and return top-k chunk dicts.

    Scoped strictly to document_id. Returns list of chunk dicts with
    an added 'bm25_score' field.
    Raises RetrievalError on failure.
    """
    k = top_k or settings.retrieval_top_k
    _ensure_index(document_id)

    index = _indexes[document_id]
    chunks = _chunk_registry[document_id]

    try:
        tokenized_query = _tokenize(query_text)
        scores = index.get_scores(tokenized_query)

        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results = []
        for idx, score in scored:
            if score > 0:  # Skip zero-score results
                chunk = dict(chunks[idx])
                chunk["bm25_score"] = float(score)
                results.append(chunk)

        logger.info("BM25 returned %d results for document_id=%s.", len(results), document_id)
        return results
    except Exception as exc:
        raise RetrievalError(f"BM25 query failed for document_id={document_id}: {exc}") from exc
