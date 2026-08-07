"""Embedding generation using Cohere API (embed-english-v3.0).

Replaces the local sentence-transformers model and OpenAI model to save RAM
and allow free development.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
import httpx

from app.config import settings
from app.exceptions import EmbeddingError

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EmbeddingError("COHERE_API_KEY is not set in the .env file. Cannot generate embeddings.")
    return api_key


def embed_texts(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embed a list of strings using Cohere's API.
    
    input_type can be 'search_document' or 'search_query'.
    Raises EmbeddingError on failure.
    """
    if not texts:
        return []

    api_key = _get_api_key()
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "embed-english-v3.0",
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"]
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.cohere.com/v2/embed",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]["float"]
            
    except Exception as exc:
        raise EmbeddingError(f"Cohere Embedding generation failed: {exc}") from exc


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed the content of a list of Chunk objects."""
    texts = [c.content for c in chunks]
    logger.info("Embedding %d chunks via Cohere API.", len(texts))
    vectors = embed_texts(texts, input_type="search_document")
    logger.info("Embedding complete.")
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    if not query or not query.strip():
        raise EmbeddingError("Cannot embed an empty query.")
    return embed_texts([query], input_type="search_query")[0]
