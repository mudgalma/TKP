"""Phase 1 end-to-end verification script.

Tests: happy path, retrieval failure (irrelevant question), and LLM timeout mock.
Run with: uv run python test_phase1.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_phase1")

# ── Locate sample PDF ────────────────────────────────────────────────────────

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "initial_tkp.pdf")

if not os.path.exists(SAMPLE_PDF):
    logger.error("Sample PDF not found at %s. Place a PDF there and re-run.", SAMPLE_PDF)
    sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pdf_bytes() -> bytes:
    with open(SAMPLE_PDF, "rb") as f:
        return f.read()


async def run_ingestion(pdf_bytes: bytes, doc_id: str) -> int:
    """Parse → chunk → embed → upsert → build BM25. Returns chunk count."""
    from app.ingestion.chunker import chunk_document
    from app.ingestion.parser import parse_pdf
    from app.rag import bm25_index, vector_store
    from app.rag.embed import embed_chunks

    logger.info("=== Ingestion started for doc_id=%s ===", doc_id)

    page_blocks = await parse_pdf(pdf_bytes, hint="mostly_text", filename="initial_tkp.pdf")
    logger.info("Parsed: %d page blocks.", len(page_blocks))

    chunks = chunk_document(page_blocks, document_id=doc_id, is_markdown=False)
    logger.info("Chunked: %d chunks.", len(chunks))

    # Validate page metadata is present
    for c in chunks:
        assert c.page_start >= 1, f"page_start missing on chunk {c.chunk_id}"
        assert c.page_end >= c.page_start, f"page_end < page_start on chunk {c.chunk_id}"

    embeddings = embed_chunks(chunks)
    logger.info("Embedded: %d vectors.", len(embeddings))

    vector_store.upsert_chunks(chunks, embeddings)
    bm25_index.build_index(doc_id, [c.__dict__ | {"content": c.content} for c in chunks])

    logger.info("=== Ingestion complete. %d chunks stored. ===", len(chunks))
    return len(chunks)


async def test_happy_path(doc_id: str) -> None:
    logger.info("\n--- TEST 1: Happy Path ---")
    from app.rag.generate import generate_answer
    from app.rag.reranker import rerank
    from app.rag.retriever import retrieve

    query = "What is the main topic of this document?"
    candidates = retrieve(query, document_id=doc_id)
    logger.info("Retrieved %d candidates.", len(candidates))
    assert candidates, "Expected at least 1 candidate from retrieval"

    ranked = rerank(query, candidates)
    logger.info("Ranked %d chunks above threshold.", len(ranked))

    result = generate_answer(query, ranked)
    logger.info("Answer: %s", result.answer[:300])
    logger.info("Citations: %s", [f"{c.document_id}:{c.page_citation}:{c.section_title}" for c in result.citations])

    assert result.answer, "Expected non-empty answer"
    logger.info("✅ Happy path passed.")


async def test_not_found_fallback(doc_id: str) -> None:
    logger.info("\n--- TEST 2: Low-Relevance Fallback (irrelevant question) ---")
    from app.rag.generate import generate_answer
    from app.rag.reranker import rerank
    from app.rag.retriever import retrieve

    query = "What is the recipe for chocolate cake with extra sprinkles?"
    candidates = retrieve(query, document_id=doc_id)
    ranked = rerank(query, candidates, min_score=0.99)  # Very high threshold → should be empty

    result = generate_answer(query, ranked)
    logger.info("Answer: %s", result.answer)
    assert "don't have enough information" in result.answer.lower(), (
        f"Expected fallback message, got: {result.answer}"
    )
    logger.info("✅ Not-found fallback passed.")


async def test_llm_timeout_mock(doc_id: str) -> None:
    logger.info("\n--- TEST 3: LLM Timeout Mock ---")
    from unittest.mock import MagicMock
    from openai import APITimeoutError
    from app.exceptions import GenerationError
    from app.rag.generate import generate_answer
    from app.rag.reranker import RankedChunk

    # Build a fake ranked chunk so we actually hit the LLM call
    fake_chunk = RankedChunk(
        chunk_id="fake-001",
        document_id=doc_id,
        content="Photosynthesis is a process used by plants.",
        page_start=1,
        page_end=1,
        section_title="Photosynthesis",
        token_count=10,
        score=0.95,
    )

    # APITimeoutError requires a `request` arg — provide a dummy
    dummy_request = MagicMock()
    timeout_error = APITimeoutError(request=dummy_request)

    with patch("app.rag.generate._call_llm", side_effect=timeout_error):
        try:
            generate_answer("What is photosynthesis?", [fake_chunk])
            assert False, "Expected GenerationError to be raised"
        except GenerationError as exc:
            logger.info("Caught expected GenerationError: %s", exc)
            logger.info("✅ Timeout mock test passed.")


async def main() -> None:
    doc_id = str(uuid.uuid4())
    pdf_bytes = load_pdf_bytes()

    chunk_count = await run_ingestion(pdf_bytes, doc_id)
    assert chunk_count > 0, "Ingestion produced 0 chunks"

    await test_happy_path(doc_id)
    await test_not_found_fallback(doc_id)
    await test_llm_timeout_mock(doc_id)

    logger.info("\n✅ All Phase 1 tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
