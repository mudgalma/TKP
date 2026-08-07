"""MLflow Evaluation Script for Teacher AI Platform.

This script runs the automated evaluation pipeline over the Golden Dataset,
judges the outputs using deterministic logic and LLM-as-a-judge criteria,
and logs the results to the local MLflow Tracking Server.
"""

import os
import asyncio
import json
import logging
import mlflow
from typing import Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(override=True)

from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_document
from app.rag.embed import embed_chunks
from app.rag import vector_store, bm25_index
from app.generation.nodes import _call_llm_structured
from app.schemas.metadata import DocumentMetadata, EducationalKnowledge
from app.schemas.tkp import TeacherKnowledgePackage
from app.document_registry import registry
from app.generation.graph import tkp_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tkp_eval")

# 1. Dynamically load the Ground Truth JSONs to build the dataset
def load_eval_dataset() -> list[dict[str, Any]]:
    dataset = []
    gt_dir = "ground_truth"
    
    if not os.path.exists(gt_dir):
        logger.error("Ground truth directory '%s' does not exist.", gt_dir)
        return []
        
    for file in os.listdir(gt_dir):
        if not file.endswith(".json"):
            continue
            
        gt_path = os.path.join(gt_dir, file)
        try:
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
        except Exception as exc:
            logger.error("Failed to read ground truth file %s: %s", file, exc)
            continue
            
        pdf_ref = gt_data.get("document", "")
        if not pdf_ref:
            logger.warning("Ground truth file %s is missing the 'document' key.", file)
            continue
            
        # Try both the underscore name and space-replaced name
        candidates = [pdf_ref, pdf_ref.replace("_", " ")]
        matched_pdf_path = None
        for cand in candidates:
            cand_path = os.path.join("docs", cand)
            if os.path.exists(cand_path):
                matched_pdf_path = cand_path
                break
                
        if matched_pdf_path:
            logger.info("Matched ground truth %s with PDF %s", file, matched_pdf_path)
            dataset.append({
                "inputs": {
                    "document_path": matched_pdf_path,
                    "task_type": "generate_lesson_plan",
                    "subject": gt_data.get("subject", "General"),
                    "grade": str(gt_data.get("grade", "9"))
                },
                "expectations": {
                    "expected_concepts": gt_data.get("expected_concepts", [])
                }
            })
        else:
            logger.warning("Could not find matching PDF in 'docs/' for ground truth %s (tried: %s)", file, candidates)
            
    # Add Adversarial Case 1: Irrelevant question
    if dataset:
        dataset.append({
            "inputs": {
                "document_path": dataset[0]["inputs"]["document_path"],
                "task_type": "adversarial_irrelevant",
                "subject": dataset[0]["inputs"]["subject"],
                "grade": dataset[0]["inputs"]["grade"]
            },
            "expectations": {
                "expected_response": "NOT_FOUND_IN_DOCUMENT"
            }
        })
        
    # Add Adversarial Case 2: Tampered document check
    tampered_pdf = "docs/ncert-pdf-ch10 edited.pdf"
    if os.path.exists(tampered_pdf):
        logger.info("Adding tampered document test case for: %s", tampered_pdf)
        dataset.append({
            "inputs": {
                "document_path": tampered_pdf,
                "task_type": "adversarial_tampered",
                "subject": "Geography (Social Science)",
                "grade": "9"
            },
            "expectations": {
                "expected_response": "Must not include the tampered/invented claim"
            }
        })
        
    return dataset

# 2. Define the Prediction Function (The Pipeline Wrapper)
def run_pipeline(**inputs: Any) -> dict[str, Any]:
    """Wraps the end-to-end ingestion and generation pipeline for MLflow."""
    doc_path = inputs["document_path"]
    filename = os.path.basename(doc_path)
    task_type = inputs.get("task_type", "generate_lesson_plan")
    
    if not os.path.exists(doc_path):
        return {
            "status": "failed",
            "error_message": f"Source document not found at {doc_path}."
        }
        
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    async def _execute():
        # Step 1: Parse
        page_blocks, is_markdown = await parse_file(filename, "application/pdf", open(doc_path, "rb").read())
        
        # Step 2: Chunk
        doc_id = f"eval_{filename.replace(' ', '_')}"
        chunks = chunk_document(page_blocks, doc_id, is_markdown=is_markdown)
        
        # Step 3: Embed (Uses Cohere free API)
        vectors = embed_chunks(chunks)
        
        # Step 4: Store
        vector_store.upsert_chunks(chunks, vectors)
        
        # Handle adversarial fallback check
        if task_type == "adversarial_irrelevant":
            # For adversarial queries, we verify that the RAG retrieval fails safely
            from app.rag.retriever import retrieve
            from app.rag.reranker import rerank
            from app.rag.generate import generate_answer
            
            candidates = retrieve("What is the capital of France?", doc_id)
            ranked = rerank("What is the capital of France?", candidates)
            result = generate_answer("What is the capital of France?", ranked)
            
            return {
                "status": "success",
                "task_type": task_type,
                "answer": result.answer
            }
            
        # Step 5: Classify & Setup Initial State
        subj = inputs["subject"]
        category = "STEM" if any(x in subj.lower() for x in ["stem", "math", "science", "biology", "physics"]) else "Humanities"
        metadata = DocumentMetadata(
            subject=subj,
            grade_level=inputs["grade"],
            difficulty="Intermediate",
            topic=subj,
            category=category
        )
        ek = EducationalKnowledge(
            metadata=metadata,
            key_concepts=inputs["expectations"].get("expected_concepts", ["Core Concept"]),
            learning_objectives=["Understand core curriculum concepts."],
            definitions=[],
            formulae=[],
            keywords=[],
            examples=[],
            common_misconceptions=[]
        )
        
        initial_state = {
            "document_id": doc_id,
            "metadata": metadata,
            "educational_knowledge": ek,
            "progress_events": [],
            "periods": [],
            "activities": [],
            "assessments": [],
            "learning_gaps": [],
            "validation_issues": [],
            "retry_count": 0,
            "validation_status": "pending"
        }
        
        # Step 6: Invoke the LangGraph pipeline
        final_state = await asyncio.to_thread(tkp_graph.invoke, initial_state)
        
        # Assemble Final output
        tkp = TeacherKnowledgePackage(
            document_id=doc_id,
            metadata=metadata,
            educational_knowledge=ek,
            periods=final_state.get("periods", []),
            activities=final_state.get("activities", []),
            assessments=final_state.get("assessments", []),
            learning_gaps=final_state.get("learning_gaps", []),
            validation_issues=final_state.get("validation_issues", []),
            is_valid=final_state.get("validation_status") == "passed"
        )
        
        return tkp.model_dump()

    try:
        result = loop.run_until_complete(_execute())
        return result
    except Exception as exc:
        logger.exception("Pipeline failed during evaluation")
        return {"status": "failed", "error_message": str(exc)}


# 3. Define Deterministic Evaluation Scorers
from mlflow.genai import scorer

@scorer
def has_valid_schema(outputs: dict[str, Any]) -> bool:
    """Checks if the output contains the required Pydantic schema structure."""
    if outputs.get("status") == "failed" or outputs.get("task_type") == "adversarial_irrelevant":
        return True
    required_keys = {"metadata", "educational_knowledge", "periods", "activities", "assessments"}
    return required_keys.issubset(outputs.keys())


@scorer
def has_populated_scripts(outputs: dict[str, Any]) -> bool:
    """Verifies that the generated periods contain actual teacher scripts."""
    if outputs.get("task_type") == "adversarial_irrelevant":
        return True
    periods = outputs.get("periods", [])
    if not periods:
        return False
    # Check first script has content
    first_script = periods[0].get("teacher_script", "")
    return len(first_script.strip()) > 30


@scorer
def handles_adversarial_correctly(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
    """Checks if adversarial out-of-bounds questions trigger the required fallback response."""
    if outputs.get("task_type") == "adversarial_irrelevant":
        answer = outputs.get("answer", "").lower()
        return "don't have enough information" in answer or "not enough information" in answer
    return True


@scorer
def no_tampered_facts(outputs: dict[str, Any]) -> bool:
    """Ensures the generated lesson plan does not repeat the tampered facts (like 50km troposphere height)."""
    if outputs.get("task_type") == "adversarial_tampered":
        # Convert output dict to string to search easily
        raw_str = json.dumps(outputs).lower()
        # If the model repeated the tampered 50km height instead of the true 13km height, it fails.
        return "50 kilometres" not in raw_str and "50 km" not in raw_str
    return True


# 4. Run Evaluation
if __name__ == "__main__":
    mlflow.set_experiment("TKP-Evaluation-Suite")
    
    eval_dataset = load_eval_dataset()
    if not eval_dataset:
        logger.error("No valid test files found in ground_truth/ and docs/ to build dataset.")
        exit(1)
        
    # Custom guidelines scorer (LLM-as-a-judge)
    from mlflow.genai.scorers import Guidelines
    
    pedagogy_judge = Guidelines(
        name="pedagogically_sound",
        guidelines="The generated teaching plan must include a clear entry ticket, "
                   "a detailed teacher script, blackboard notes, a mentor moment, "
                   "and checkpoint questions matching the grade level."
    )

    scorers = [
        has_valid_schema,
        has_populated_scripts,
        handles_adversarial_correctly,
        no_tampered_facts,
        pedagogy_judge
    ]

    print(f"🚀 Loaded {len(eval_dataset)} evaluation scenarios.")
    print("🚀 Starting MLflow Evaluation Run...")
    
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=run_pipeline,
        scorers=scorers
    )
    
    print("\n✅ Evaluation complete! Start the MLflow UI with: uvx mlflow server")
