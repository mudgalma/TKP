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

logging.basicConfig(level=logging.INFO)
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
