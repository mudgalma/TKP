"""Document chunking utilities.

Splits parsed page blocks into semantically meaningful chunks, attaches
full provenance metadata (including page ranges for accurate citations),
and assigns deterministic chunk IDs for idempotent upserts.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.exceptions import ChunkingError
from app.ingestion.parser import PageBlock

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Chunk model                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""

    chunk_id: str        # Deterministic hash: md5(document_id + chunk_index)
    document_id: str     # FK → parent document
    chunk_index: int     # Ordering within the document
    content: str         # Raw text (what gets embedded and shown as citation snippet)
    page_start: int      # First page this chunk spans
    page_end: int        # Last page this chunk spans (may equal page_start)
    section_title: str   # Nearest heading, e.g. "Chapter 3 > Photosynthesis"
    token_count: int     # Approximate token count (chars // 4)
    chunk_type: str = "text"  # "text" | "table" | "equation" | "figure_caption"

    @property
    def page_citation(self) -> str:
        """Human-readable page range for citations."""
        if self.page_start == self.page_end:
            return f"p{self.page_start}"
        return f"p{self.page_start}-{self.page_end}"

    def to_chroma_metadata(self) -> dict:
        """Serialise to flat dict for Chroma metadata storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
            "token_count": self.token_count,
            "chunk_type": self.chunk_type,
        }


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a deterministic chunk ID — same inputs always produce the same ID."""
    raw = f"{document_id}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _approx_tokens(text: str) -> int:
    """Estimate token count as characters ÷ 4 (good enough for budgeting)."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
#  Heading detection                                                           #
# --------------------------------------------------------------------------- #

# Regex patterns that suggest a line is a heading (for PyMuPDF plain text)
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:Chapter|Section|Unit|Part|Module)\s+[\dIVXivx]+[.\s]"  # explicit keywords
    r"|[A-Z][A-Z\s]{3,40}$"                                      # ALL CAPS short line
    r"|(?:\d+\.)+\d?\s+[A-Z]"                                    # numbered: 1.2 Title
    r")"
)


def _detect_heading(line: str) -> bool:
    """Return True if line looks like a section heading in plain text."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(_HEADING_RE.match(stripped))


def _extract_markdown_heading(line: str) -> str | None:
    """Return heading text if line is a markdown ATX heading, else None."""
    match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
    return match.group(1).strip() if match else None


# --------------------------------------------------------------------------- #
#  Splitters                                                                   #
# --------------------------------------------------------------------------- #

def _split_markdown_by_headers(
    page_blocks: list[PageBlock],
    document_id: str,
) -> list[Chunk]:
    """Chunk LlamaParse markdown output by header boundaries.

    Tracks which pages each chunk spans so citations can cover page ranges.
    """
    chunks: list[Chunk] = []
    chunk_index = 0
    current_lines: list[str] = []
    current_pages: list[int] = []
    current_heading = "Introduction"

    def _flush() -> None:
        nonlocal chunk_index
        text = "\n".join(current_lines).strip()
        if not text:
            return
        pages = sorted(set(current_pages)) if current_pages else [1]
        chunk = Chunk(
            chunk_id=_make_chunk_id(document_id, chunk_index),
            document_id=document_id,
            chunk_index=chunk_index,
            content=text,
            page_start=pages[0],
            page_end=pages[-1],
            section_title=current_heading,
            token_count=_approx_tokens(text),
        )
        chunks.append(chunk)
        chunk_index += 1

    for page_num, text in page_blocks:
        for line in text.splitlines():
            heading = _extract_markdown_heading(line)
            if heading:
                _flush()
                current_lines = []
                current_pages = []
                current_heading = heading
            current_lines.append(line)
            current_pages.append(page_num)

    _flush()
    return chunks


def _split_plain_text(
    page_blocks: list[PageBlock],
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk PyMuPDF plain text output using a sliding window.

    Respects paragraph boundaries where possible and tracks page spans.
    As it iterates, it watches for heading-like lines and stamps every
    chunk with the most recently seen heading as section_title.
    """
    # Flatten page blocks into tagged paragraphs
    tagged: list[tuple[str, int]] = []  # (paragraph, page_num)

    for page_num, text in page_blocks:
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if not para:
                continue
            tagged.append((para, page_num))

    chunks: list[Chunk] = []
    chunk_index = 0

    # Sliding window state
    buffer_paras: list[str] = []
    buffer_pages: list[int] = []
    buffer_tokens = 0
    # Track which heading was active when the current buffer started filling
    active_section = "Introduction"

    def _flush() -> None:
        nonlocal chunk_index
        text = "\n\n".join(buffer_paras).strip()
        if not text:
            return
        pages = sorted(set(buffer_pages))
        chunk = Chunk(
            chunk_id=_make_chunk_id(document_id, chunk_index),
            document_id=document_id,
            chunk_index=chunk_index,
            content=text,
            page_start=pages[0],
            page_end=pages[-1],
            section_title=active_section,
            token_count=_approx_tokens(text),
        )
        chunks.append(chunk)

    for para, page_num in tagged:
        # Detect if this paragraph is a heading — but don't update active_section
        # until AFTER we flush the previous buffer, so the old chunk keeps the
        # old heading.
        is_heading = _detect_heading(para)

        para_tokens = _approx_tokens(para)
        if buffer_tokens + para_tokens > chunk_size and buffer_paras:
            _flush()
            chunk_index += 1
            # Overlap: keep the last overlap-worth of paragraphs
            overlap_buf: list[str] = []
            overlap_pages: list[int] = []
            overlap_tokens = 0
            for p, pg in zip(reversed(buffer_paras), reversed(buffer_pages)):
                if overlap_tokens + _approx_tokens(p) > chunk_overlap:
                    break
                overlap_buf.insert(0, p)
                overlap_pages.insert(0, pg)
                overlap_tokens += _approx_tokens(p)
            buffer_paras = overlap_buf
            buffer_pages = overlap_pages
            buffer_tokens = overlap_tokens

        # NOW update the section title — this ensures that if a heading
        # triggered a flush above, the flushed chunk kept the old title,
        # and the new buffer starts with the new title.
        if is_heading:
            active_section = para

        buffer_paras.append(para)
        buffer_pages.append(page_num)
        buffer_tokens += para_tokens

    if buffer_paras:
        _flush()

    return chunks


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def chunk_document(
    page_blocks: list[PageBlock],
    document_id: str,
    is_markdown: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split parsed page blocks into annotated chunks.

    Args:
        page_blocks:   Output from parser.parse_file — list of (page_num, text).
        document_id:   Stable UUID for the parent document.
        is_markdown:   True when input came from LlamaParse (has ## headers).
        chunk_size:    Target max tokens per chunk (defaults to settings value).
        chunk_overlap: Overlap tokens between adjacent chunks.

    Returns:
        List of Chunk objects ready for embedding and vector store upsert.

    Raises:
        ChunkingError if chunking fails or produces no chunks.
    """
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    if not page_blocks:
        raise ChunkingError(f"No page blocks to chunk for document_id={document_id}")

    try:
        if is_markdown:
            logger.info("Chunking using MarkdownHeaderSplitter (LlamaParse input).")
            chunks = _split_markdown_by_headers(page_blocks, document_id)
        else:
            logger.info("Chunking using RecursiveCharacterTextSplitter (PyMuPDF input).")
            chunks = _split_plain_text(page_blocks, document_id, cs, co)
    except Exception as exc:
        raise ChunkingError(f"Chunking failed for document_id={document_id}: {exc}") from exc

    if not chunks:
        raise ChunkingError(f"Chunking produced zero chunks for document_id={document_id}")

    logger.info(
        "Chunked document_id=%s into %d chunks (is_markdown=%s).",
        document_id,
        len(chunks),
        is_markdown,
    )
    return chunks
