from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A structured reference to a source chunk."""
    document_id: str
    chunk_id: str
    page_range: str
    section_title: str


class ValidationIssue(BaseModel):
    """A structured issue returned by the validator node."""
    target: str = Field(..., description="The generator that caused the issue, e.g., 'assessment_generator'")
    code: str = Field(..., description="Error code, e.g., 'UNSUPPORTED_CLAIM' or 'MISSING_ANSWER_KEY'")
    message: str = Field(..., description="Human readable message")
    period_id: Optional[str] = Field(default=None, description="The period ID if applicable")
    citation: Optional[Citation] = Field(default=None, description="The citation related to the issue, if any")
