import operator
from typing import Annotated, List, Optional, TypedDict

from app.schemas.metadata import DocumentMetadata, EducationalKnowledge
from app.schemas.period import TeachingPeriod
from app.schemas.activity import Activity
from app.schemas.assessment import Assessment
from app.schemas.gap import LearningGap
from app.schemas.core import ValidationIssue


def merge_lists(a: List, b: List) -> List:
    """Reducer to append lists together instead of overwriting."""
    if a is None:
        return b if b else []
    if b is None:
        return a
    return a + b


class TKPState(TypedDict):
    """The shared state for the LangGraph multi-agent orchestration."""
    document_id: str
    
    # Base Knowledge
    metadata: Optional[DocumentMetadata]
    educational_knowledge: Optional[EducationalKnowledge]
    
    # Progress streaming
    progress_events: Annotated[List[str], merge_lists]
    
    # Generated content (Parallel arrays need reducers)
    periods: Annotated[List[TeachingPeriod], merge_lists]
    activities: Annotated[List[Activity], merge_lists]
    assessments: Annotated[List[Assessment], merge_lists]
    learning_gaps: Annotated[List[LearningGap], merge_lists]
    
    # Validation state
    validation_issues: Annotated[List[ValidationIssue], merge_lists]
    retry_count: int
    validation_status: str  # "pending", "passed", "failed"
