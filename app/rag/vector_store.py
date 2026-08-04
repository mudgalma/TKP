"""ChromaDB persistent vector store wrapper.

Chroma is the single source of truth for chunk content and metadata in Phase 1.
All queries are scoped by document_id to prevent cross-document leakage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.exceptions import RetrievalError

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Lazy singleton client + collection
_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    """Return (lazy-init) the persistent Chroma collection."""
    global _client, _collection
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            _collection = _client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready at '%s'.",
                settings.chroma_collection_name,
                settings.chroma_persist_dir,
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to initialise ChromaDB: {exc}") from exc
    return _collection


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert chunks into ChromaDB.

    Uses deterministic chunk_id so re-runs overwrite instead of duplicating.
    Raises RetrievalError on failure.
    """
    if not chunks:
        return

    collection = _get_collection()
    try:
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.to_chroma_metadata() for c in chunks],
        )
        logger.info("Upserted %d chunks into ChromaDB.", len(chunks))
    except Exception as exc:
        raise RetrievalError(f"ChromaDB upsert failed: {exc}") from exc


def query_chunks(
    query_embedding: list[float],
    document_id: str,
    top_k: int | None = None,
) -> list[dict]:
    """Query Chroma for the top-k most similar chunks, scoped to document_id.

    Returns a list of dicts with keys: chunk_id, document_id, content,
    page_start, page_end, section_title, token_count, score.
    Raises RetrievalError on failure.
    """
    k = top_k or settings.retrieval_top_k
    collection = _get_collection()

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            where={"document_id": document_id},  # strict isolation
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RetrievalError(f"ChromaDB query failed: {exc}") from exc

    hits = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({
            "chunk_id": meta.get("chunk_id", ""),
            "document_id": meta.get("document_id", ""),
            "content": doc,
            "page_start": meta.get("page_start", 1),
            "page_end": meta.get("page_end", 1),
            "section_title": meta.get("section_title", ""),
            "token_count": meta.get("token_count", 0),
            "score": 1.0 - dist,  # convert cosine distance to similarity
        })

    logger.info("ChromaDB returned %d hits for document_id=%s.", len(hits), document_id)
    return hits


def delete_document(document_id: str) -> None:
    """Remove all chunks for a document — used before re-ingesting."""
    collection = _get_collection()
    try:
        collection.delete(where={"document_id": document_id})
        logger.info("Deleted all chunks for document_id=%s.", document_id)
    except Exception as exc:
        raise RetrievalError(f"Failed to delete chunks for document_id={document_id}: {exc}") from exc


def get_all_chunks_for_document(document_id: str) -> list[dict]:
    """Fetch all chunk metadata for a document (used to rebuild BM25 index)."""
    collection = _get_collection()
    try:
        result = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        chunks = []
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
            chunks.append({**meta, "content": doc})
        return chunks
    except Exception as exc:
        raise RetrievalError(f"Failed to fetch chunks for document_id={document_id}: {exc}") from exc
