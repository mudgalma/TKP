from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.metadata import DocumentMetadata, EducationalKnowledge
from app.schemas.period import TeachingPeriod
from app.schemas.activity import Activity
from app.schemas.assessment import Assessment
from app.schemas.gap import LearningGap
from app.schemas.core import Citation, ValidationIssue

class TeacherKnowledgePackage(BaseModel):
    """The canonical output schema for the entire package."""
    document_id: str
    metadata: DocumentMetadata
    educational_knowledge: EducationalKnowledge
    periods: List[TeachingPeriod] = Field(default_factory=list)
    activities: List[Activity] = Field(default_factory=list)
    assessments: List[Assessment] = Field(default_factory=list)
    learning_gaps: List[LearningGap] = Field(default_factory=list)
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    is_valid: bool = Field(default=False)
