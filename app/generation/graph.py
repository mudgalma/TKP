from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.generation.state import TKPState
from app.generation.nodes import (
    educational_extractor,
    teaching_planner,
    content_generator,
    activity_generator,
    assessment_generator,
    learning_gap_analyzer,
    package_assembler,
    validator
)

def check_validation(state: TKPState) -> Literal["end", "retry"]:
    """Conditional edge router based on validation status."""
    status = state.get("validation_status", "pending")
    if status == "passed" or status == "failed":
        return "end"
    return "retry"

def build_graph():
    """Build the TKP LangGraph orchestration workflow."""
    workflow = StateGraph(TKPState)
    
    # Add nodes
    workflow.add_node("educational_extractor", educational_extractor)
    workflow.add_node("teaching_planner", teaching_planner)
    
    # We will currently run generators sequentially for reliability, 
    # as parallelization of all 4 at once can hit rate limits or context issues.
    workflow.add_node("content_generator", content_generator)
    workflow.add_node("activity_generator", activity_generator)
    workflow.add_node("assessment_generator", assessment_generator)
    workflow.add_node("learning_gap_analyzer", learning_gap_analyzer)
    
    workflow.add_node("package_assembler", package_assembler)
    workflow.add_node("validator", validator)
    
    # Add edges
    workflow.add_edge(START, "educational_extractor")
    workflow.add_edge("educational_extractor", "teaching_planner")
    
    # Sequential generation
    workflow.add_edge("teaching_planner", "content_generator")
    workflow.add_edge("content_generator", "activity_generator")
    workflow.add_edge("activity_generator", "assessment_generator")
    workflow.add_edge("assessment_generator", "learning_gap_analyzer")
    workflow.add_edge("learning_gap_analyzer", "package_assembler")
    
    workflow.add_edge("package_assembler", "validator")
    
    # Conditional retry routing
    workflow.add_conditional_edges(
        "validator",
        check_validation,
        {
            "end": END,
            "retry": "content_generator"  # Loop back to fix issues
        }
    )
    
    return workflow.compile()

tkp_graph = build_graph()
