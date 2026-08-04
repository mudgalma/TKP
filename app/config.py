"""Centralized configuration for the Teacher AI Platform.

All API keys, model names, thresholds, and tunables live here.
Never import secrets directly in logic files — always go through this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return env var or raise at startup with a clear message."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Please add it to your .env file."
        )
    return value


@dataclass(frozen=True)
class Config:
    # ── API Keys ────────────────────────────────────────────────────────────
    openrouter_api_key: str = field(default_factory=lambda: _require("OPENROUTER_API_KEY"))
    llama_cloud_api_key: str = field(default_factory=lambda: os.environ.get("LLAMA_CLOUD_API_KEY", ""))

    # ── OpenRouter / LLM ────────────────────────────────────────────────────
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct"))
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_batch_size: int = 32

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection_name: str = "tkp_chunks"

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 512          # tokens — for RecursiveCharacterTextSplitter
    chunk_overlap: int = 64
    max_context_tokens: int = 6000  # hard budget sent to the LLM

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval_top_k: int = 6       # candidates from each retriever (vector & BM25)
    rerank_top_k: int = 5          # chunks passed to the LLM after reranking
    rerank_min_score: float = 0.3  # sigmoid-normalized; below → "not found" fallback

    # ── Parsing ─────────────────────────────────────────────────────────────
    max_upload_bytes: int = 50 * 1024 * 1024   # 50 MB
    max_question_length: int = 1000

    # ── Reranker ────────────────────────────────────────────────────────────
    reranker_model: str = field(default_factory=lambda: os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base"))


# Single global instance — import this everywhere
settings = Config()
