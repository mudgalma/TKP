"""Embedding generation using a local sentence-transformers model.

Uses BAAI/bge-small-en-v1.5 by default — $0 cost, runs on CPU.
Embeds in configurable batches for throughput.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import settings
from app.exceptions import EmbeddingError

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Lazy singleton — load the model once on first use
_model = None


def _get_model():
    """Return (and lazy-load) the sentence-transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model: {exc}") from exc
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in batches. Returns list of float vectors.

    Raises EmbeddingError on failure.
    """
    if not texts:
        return []

    model = _get_model()
    try:
        vectors = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]
    except Exception as exc:
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed the content of a list of Chunk objects.

    Returns vectors in the same order as the input chunks.
    """
    texts = [c.content for c in chunks]
    logger.info("Embedding %d chunks in batches of %d.", len(texts), settings.embedding_batch_size)
    vectors = embed_texts(texts)
    logger.info("Embedding complete.")
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    if not query or not query.strip():
        raise EmbeddingError("Cannot embed an empty query.")
    return embed_texts([query])[0]
