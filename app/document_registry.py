"""In-memory Document Registry for tracking uploads and classification results.

Provides thread-safe access to document metadata and status,
acting as a placeholder for a real database (SQLite/Supabase).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Models                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    content_type: str
    file_size_bytes: int
    hint: str
    status: str = "uploading"  # uploading, parsing, chunking, embedding, classifying, ready, error
    error_message: str | None = None
    upload_time: datetime = field(default_factory=datetime.utcnow)
    classification: dict[str, Any] | None = None  # Holds Stage 2/3 LLM extraction


@dataclass
class JobRecord:
    job_id: str
    document_id: str
    status: str = "running" # running, completed, failed
    progress_events: list[str] = field(default_factory=list)
    final_output: dict[str, Any] | None = None
    error_message: str | None = None
    start_time: datetime = field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
#  Store Implementation                                                        #
# --------------------------------------------------------------------------- #

class DocumentRegistry:
    """Thread-safe in-memory store for document records."""

    def __init__(self):
        self._lock = threading.RLock()
        self._store: dict[str, DocumentRecord] = {}

    def add_document(self, record: DocumentRecord) -> None:
        """Register a new document."""
        with self._lock:
            self._store[record.document_id] = record
            logger.info("Registered document %s: %s", record.document_id, record.filename)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Retrieve a document by ID."""
        with self._lock:
            return self._store.get(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        """List all registered documents (newest first)."""
        with self._lock:
            docs = list(self._store.values())
            docs.sort(key=lambda d: d.upload_time, reverse=True)
            return docs

    def update_status(self, document_id: str, status: str, error_message: str | None = None) -> None:
        """Update the processing status of a document."""
        with self._lock:
            if doc := self._store.get(document_id):
                doc.status = status
                if error_message:
                    doc.error_message = error_message
                logger.debug("Document %s status -> %s", document_id, status)

    def set_classification(self, document_id: str, classification: dict[str, Any]) -> None:
        """Save the extracted Stage 2/3 metadata."""
        with self._lock:
            if doc := self._store.get(document_id):
                doc.classification = classification
                logger.info("Saved classification for document %s", document_id)


# Global singleton instance
registry = DocumentRegistry()


class JobRegistry:
    """Thread-safe store for LangGraph jobs."""
    def __init__(self):
        self._lock = threading.RLock()
        self._store: dict[str, JobRecord] = {}
        
    def create_job(self, job_id: str, document_id: str) -> JobRecord:
        with self._lock:
            job = JobRecord(job_id=job_id, document_id=document_id)
            self._store[job_id] = job
            return job
            
    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._store.get(job_id)
            
    def update_job(self, job_id: str, status: str, final_output: dict | None = None, error: str | None = None):
        with self._lock:
            if job := self._store.get(job_id):
                job.status = status
                if final_output:
                    job.final_output = final_output
                if error:
                    job.error_message = error

    def append_event(self, job_id: str, event: str):
        with self._lock:
            if job := self._store.get(job_id):
                job.progress_events.append(event)


job_registry = JobRegistry()
