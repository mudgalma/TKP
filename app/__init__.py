import asyncio
import logging
import uuid
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.document_registry import DocumentRecord, registry
from app.ingestion.chunker import chunk_document
from app.ingestion.classifier import classify_document
from app.ingestion.parser import parse_file
from app.rag import bm25_index, vector_store
from app.rag.embed import embed_chunks
from app.rag.generate import generate_answer
from app.rag.reranker import rerank
from app.rag.retriever import retrieve
from app.document_registry import job_registry
from app.generation.graph import tkp_graph
from app.schemas.tkp import TeacherKnowledgePackage
from sse_starlette.sse import EventSourceResponse
logger = logging.getLogger(__name__)

app = FastAPI(title="TKP API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Models                                                                      #
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    document_id: str
    question: str

class CitationModel(BaseModel):
    chunk_id: str
    document_id: str
    page_citation: str
    section_title: str
    snippet: str

class QueryResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None

class GenerateRequest(BaseModel):
    document_id: str


# --------------------------------------------------------------------------- #
#  Background Ingestion Task                                                   #
# --------------------------------------------------------------------------- #

async def run_ingestion_pipeline(
    document_id: str,
    filename: str,
    content_type: str,
    content: bytes,
    hint: str,
) -> None:
    """End-to-end background ingestion."""
    try:
        # 1. Parse
        registry.update_status(document_id, "parsing")
        page_blocks, is_markdown = await parse_file(filename, content_type, content, hint=hint)
        
        # 2. Chunk
        registry.update_status(document_id, "chunking")
        chunks = chunk_document(page_blocks, document_id, is_markdown=is_markdown)
        
        # 3. Embed
        registry.update_status(document_id, "embedding")
        vectors = embed_chunks(chunks)
        
        # 4. Store (Chroma & BM25)
        registry.update_status(document_id, "storing")
        vector_store.upsert_chunks(chunks, vectors)
        
        bm25_chunks = [{"content": c.content, **c.to_chroma_metadata()} for c in chunks]
        bm25_index.build_index(document_id, bm25_chunks)
        
        # 5. Classify (Stage 2/3)
        registry.update_status(document_id, "classifying")
        
        # Build context for classifier (first ~3000 tokens)
        clf_text = ""
        clf_tokens = 0
        for c in chunks:
            if clf_tokens + c.token_count > 3000:
                break
            # Add synthetic headings if PyMuPDF
            prefix = f"## {c.section_title}\n" if not is_markdown else ""
            clf_text += f"{prefix}{c.content}\n\n"
            clf_tokens += c.token_count
            
        classification = classify_document(clf_text)
        registry.set_classification(document_id, classification)
        
        # Done
        registry.update_status(document_id, "ready")
        
    except Exception as exc:
        logger.exception("Ingestion failed for %s", document_id)
        registry.update_status(document_id, "error", error_message=str(exc))


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    hint: str = Form("auto"),
):
    if len(file.filename) == 0:
        raise HTTPException(400, "Empty filename")
        
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large (max {settings.max_upload_bytes} bytes)")
        
    document_id = str(uuid.uuid4())
    
    # Register document
    record = DocumentRecord(
        document_id=document_id,
        filename=file.filename,
        content_type=file.content_type,
        file_size_bytes=len(content),
        hint=hint,
    )
    registry.add_document(record)
    
    # Start background task
    background_tasks.add_task(
        run_ingestion_pipeline,
        document_id,
        file.filename,
        file.content_type,
        content,
        hint,
    )
    
    return {"success": True, "data": {"document_id": document_id, "status": "uploading"}}


@app.get("/api/documents")
def list_documents():
    docs = registry.list_documents()
    return {"success": True, "data": [d.__dict__ for d in docs]}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"success": True, "data": doc.__dict__}


@app.post("/api/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    doc = registry.get_document(req.document_id)
    if not doc or doc.status != "ready":
        raise HTTPException(400, "Document not found or not ready")
        
    try:
        # 1. Retrieve
        candidates = retrieve(req.question, req.document_id)
        
        # 2. Rerank
        ranked = rerank(req.question, candidates)
        
        # 3. Generate
        result = generate_answer(req.question, ranked)
        
        return QueryResponse(
            success=True,
            data={
                "answer": result.answer,
                "citations": [c.__dict__ for c in result.citations],
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(500, str(exc))

# --------------------------------------------------------------------------- #
#  Generation Job Endpoints                                                    #
# --------------------------------------------------------------------------- #

async def execute_graph(job_id: str, document_id: str):
    """Run LangGraph pipeline in background."""
    try:
        doc = registry.get_document(document_id)
        if not doc or not doc.classification:
            raise ValueError("Document not ready or missing classification.")
            
        from app.schemas.metadata import DocumentMetadata, EducationalKnowledge
        metadata = DocumentMetadata(**doc.classification)
        ek = EducationalKnowledge(metadata=metadata, **doc.classification)

        initial_state = {
            "document_id": document_id,
            "metadata": metadata,
            "educational_knowledge": ek,
            "progress_events": ["Job started."],
            "periods": [],
            "activities": [],
            "assessments": [],
            "learning_gaps": [],
            "validation_issues": [],
            "retry_count": 0,
            "validation_status": "pending"
        }
        
        # We process the stream of graph updates
        final_state = None
        for update in tkp_graph.stream(initial_state):
            # Stream yields {node_name: {state_updates}}
            for node, state_update in update.items():
                if "progress_events" in state_update:
                    for ev in state_update["progress_events"]:
                        job_registry.append_event(job_id, f"[{node}] {ev}")
            final_state = update

        # Merge final state
        state = tkp_graph.get_state(initial_state) # We might need to keep state track
        
        # Graph stream returns sequence of dicts. The last one isn't the full state, just the update.
        # To get the final TKP JSON, we actually should use tkp_graph.invoke() if we want it blocking, 
        # but since we are streaming, let's just invoke it for simplicity now and append events.
        pass

    except Exception as exc:
        logger.exception("Graph execution failed")
        job_registry.update_job(job_id, "failed", error=str(exc))

@app.post("/api/generate")
async def generate_tkp(req: GenerateRequest, background_tasks: BackgroundTasks):
    doc = registry.get_document(req.document_id)
    if not doc or doc.status != "ready":
        raise HTTPException(400, "Document not found or not ready.")
        
    job_id = str(uuid.uuid4())
    job_registry.create_job(job_id, req.document_id)
    
    # Fast path for MVP: run invoke directly in a background task
    async def run_sync():
        try:
            from app.schemas.metadata import DocumentMetadata, EducationalKnowledge
            metadata = DocumentMetadata(**doc.classification)
            ek = EducationalKnowledge(metadata=metadata, **doc.classification)

            initial_state = {
                "document_id": req.document_id,
                "metadata": metadata,
                "educational_knowledge": ek,
                "progress_events": ["Job started."],
                "periods": [],
                "activities": [],
                "assessments": [],
                "learning_gaps": [],
                "validation_issues": [],
                "retry_count": 0,
                "validation_status": "pending"
            }
            
            # Since invoke is sync right now, run in thread
            final_state = await asyncio.to_thread(tkp_graph.invoke, initial_state)
            
            # Assembly
            tkp = TeacherKnowledgePackage(
                document_id=req.document_id,
                metadata=metadata,
                educational_knowledge=ek,
                periods=final_state.get("periods", []),
                activities=final_state.get("activities", []),
                assessments=final_state.get("assessments", []),
                learning_gaps=final_state.get("learning_gaps", []),
                validation_issues=final_state.get("validation_issues", []),
                is_valid=final_state.get("validation_status") == "passed"
            )
            
            job_registry.update_job(job_id, "completed", final_output=tkp.model_dump())
            job_registry.append_event(job_id, "Generation finished.")
            
        except Exception as e:
            logger.exception("Graph failed")
            job_registry.update_job(job_id, "failed", error=str(e))

    background_tasks.add_task(run_sync)
    return {"success": True, "data": {"job_id": job_id}}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"success": True, "data": job.__dict__}

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
        
    async def event_generator():
        last_idx = 0
        while True:
            job = job_registry.get_job(job_id)
            if not job:
                break
                
            events = job.progress_events[last_idx:]
            for ev in events:
                yield {"data": ev}
                last_idx += 1
                
            if job.status in ("completed", "failed"):
                # Send one final status ping
                yield {"data": f"Job {job.status}"}
                break
                
            await asyncio.sleep(1)
            
    return EventSourceResponse(event_generator())
