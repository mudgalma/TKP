"""Typed exception hierarchy for the Teacher AI Platform.

All layer-specific exceptions subclass AppError so callers can catch broadly
or narrowly depending on what they need. Raw third-party exceptions must never
leak past a module boundary.
"""

from __future__ import annotations


class AppError(Exception):
    """Base exception for the entire application."""


class ParsingError(AppError):
    """Raised when document parsing fails (PyMuPDF or LlamaParse)."""


class ChunkingError(AppError):
    """Raised when document chunking or metadata extraction fails."""


class EmbeddingError(AppError):
    """Raised when embedding generation fails."""


class RetrievalError(AppError):
    """Raised when vector store or BM25 retrieval fails."""


class RerankerError(AppError):
    """Raised when the cross-encoder reranker fails."""


class GenerationError(AppError):
    """Raised when the LLM generation call fails (timeout, bad response, etc.)."""


class ValidationError(AppError):
    """Raised when input validation fails (bad file type, size, etc.)."""


class ConfigError(AppError):
    """Raised when a required configuration or env variable is missing."""
