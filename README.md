# TKP — Teacher Knowledge Package (TKP)

A modular Python toolkit for building retrieval-augmented generation (RAG) pipelines for educational content: document ingestion & chunking, hybrid indexing (BM25 + vector), reranking, and grounded generation orchestration. TKP is intended for researchers and engineers building QA, summarization, tutoring, or lesson-generation pipelines that require explicit ingestion → retrieval → generation components.

Badges
- Python 3.9+
- Core libs: FastAPI, Streamlit (for demos), ChromaDB (vector store)

Quick highlights
- Robust document parsing for PDFs, PPTX, DOCX, and text with routing for scanned vs text docs.
- Hybrid retrieval: BM25 + vector embeddings, with reranking before LLM generation.
- Grounded, citation-aware generation using OpenRouter/OpenAI-compatible client.
- Generation orchestration: graph + node/state pattern for multi-step generation flows.
- Simple demo/frontend and an experiment harness for local testing.

Table of contents
- What this is
- Project layout
- Quick start
- Configuration (exact env vars)
- Examples (parsing & generation)
- Development notes
- Known discrepancies & optional integrations
- Contributing

What this is
TKP is a developer-friendly toolkit to convert documents into retrieval-ready chunks, index them with both sparse and dense methods, and run generation workflows that combine retrieved context with LLMs while enforcing strict context isolation and citation requirements.

Stack
- Language: Python (primary)
- Runtime / frameworks: FastAPI (backend), Streamlit (demo), uvicorn (ASGI server)
- Notable libraries (observed in pyproject.toml):
  - chromadb (vector store persistence)
  - openai (OpenRouter-compatible client usage)
  - pymupdf, python-pptx, python-docx (parsing)
  - langchain-core, langchain-text-splitters (text splitting utilities)
  - rank-bm25 (sparse retrieval baseline)
  - sentence-transformers (embedding / reranker clients)
  - tenacity (retries)

Project layout (top-level)
```
app/                    # main package: config, parsing, ingestion, rag, generation, schemas
  __init__.py
  config.py             # centralized config and environment mapping
  document_registry.py  # document metadata/registry
  parser.py             # top-level parsing utilities
  ingestion/            # parsers, chunker, classifier
  rag/                  # bm25_index, embed, retriever, reranker, vector_store, generate
  generation/           # graph, nodes, state for orchestrated generation
  schemas/              # Pydantic schemas for domain models / metadata
frontend/
  main.py               # demo / UI entry point (Streamlit or ASGI)
ground_truth/           # datasets / ground truth JSONs for evaluation
src/tkp/                # package marker / small namespace
scripts/                # developer utilities
test_phase1.py          # experiment / smoke test runner
pyproject.toml          # project metadata & declared dependencies
.env.example            # example env variables
initial_tkp.pdf         # project design brief / notes
uv.lock                 # lockfile for local toolchain
```

How it fits together
- Documents are parsed and normalized in app/ingestion/ (see parser.py, chunker.py). PDF routing chooses PyMuPDF (fitz) for text-heavy docs and llama-parse for scanned/tables/equations.
- Chunks are indexed via app/rag/ (bm25_index.py and vector_store.py). Embeddings and reranking are pluggable via settings in app/config.py.
- app/generation/ (graph.py, nodes.py, state.py) implements orchestration for multi-step generation; app/rag/generate.py constructs a context-bounded prompt and calls the LLM via an OpenAI-compatible OpenRouter client. Generation strictly enforces context-only answers and citation requirements.

Quick start — run locally
1. Clone and create a venv
```bash
git clone https://github.com/mudgalma/TKP.git
cd TKP
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

2. Install the project (editable mode recommended)
```bash
pip install -e .
```

3. Configure environment
```bash
cp .env.example .env
# Edit .env and populate keys and paths described below
```

4. Run a quick experiment
```bash
python test_phase1.py
```
This script provides an end-to-end smoke test of ingestion → retrieval → generation using included sample data.

5. Run the backend and frontend (if you want the demo)
- FastAPI backend (if an `app` FastAPI object exists in code):
```bash
uvicorn app:app --reload --port 8000
```
- Streamlit demo (frontend):
```bash
streamlit run frontend/main.py
```
If an ASGI app exists in frontend/main.py, run:
```bash
uvicorn frontend.main:app --reload --port 8501
```

Configuration — exact env vars (from app/config.py)
- OPENROUTER_API_KEY (required) — used by app/rag/generate.py to call OpenRouter-compatible APIs
- LLAMA_CLOUD_API_KEY (optional) — used by llama-parse path for scanned/tables PDFs
- EMBEDDING_MODEL (optional) — default set in app/config.py (embedding model key)
- RERANKER_MODEL (optional) — reranker model key
- CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME — paths and collection names for ChromaDB persistence

Check app/config.py for full names and defaults. The module raises a clear error at startup if required env vars (like OPENROUTER_API_KEY) are missing.

Examples

1) Parse a PDF (Python)
```python
import asyncio
from app.ingestion.parser import parse_file

# content: bytes from reading a PDF file, content_type like "application/pdf"
# This is an async API; use asyncio.run in scripts.
blocks, is_markdown = asyncio.run(parse_file("book.pdf", "application/pdf", pdf_bytes))
# blocks -> list of (page_number, text) tuples
```

2) Generate a grounded answer (Python)
- The public API is generate_answer(question, ranked_chunks) in app/rag/generate.py.
- ranked_chunks are produced by retrieval + reranker (see app/rag/reranker.py and retriever.py).
```python
from app.rag.generate import generate_answer

# ranked_chunks: list of RankedChunk (see app/rag/reranker.py)
result = generate_answer("What is X?", ranked_chunks)
print(result.answer)
for c in result.citations:
    print(c.page_citation, c.section_title)
```

3) Quick CLI-style rebuild (conceptual)
```bash
# Example placeholders — inspect app/rag/vector_store.py for exact functions
python -c "from app.rag import vector_store; vector_store.build_index('docs/')"
```

Development notes & maintainer guidance
- Ingestion:
  - `app/ingestion/parser.py` routes between PyMuPDF and llama-parse. Adjust chunk sizes and chunk_overlap in `app/config.py`.
- Retrieval:
  - `app/rag/bm25_index.py` provides a text-based baseline.
  - `app/rag/vector_store.py` and `app/rag/embed.py` are embedding + vector index hooks (plug your provider).
- Generation:
  - `app/rag/generate.py` uses an explicit system prompt and enforces strict rules (answers must come from <context> and every factual claim must have a citation). It retries only on connection/timeouts and raises GenerationError for client 4xx LLM errors.
- Schemas:
  - `app/schemas/` contains Pydantic models for consistent metadata.

Known discrepancies & optional integrations
- The prior README mentions Cohere and MLflow prominently. The current pyproject.toml does not include `cohere` or `mlflow` in dependencies. If you rely on Cohere embeddings or MLflow evaluation, add them to pyproject.toml and document expected env vars.
- The generation path uses an OpenRouter-compatible client (openai package configured with base_url). By default app/config.py points to OpenRouter; you can swap providers by changing settings and/or the client logic.

Testing & CI
- There is no tests/ folder beyond the sample `test_phase1.py`. Use `test_phase1.py` as an integration template and add unit tests under tests/ plus a basic GitHub Actions workflow to run linters and test_phase1.py for PRs.

Contributing
- Recommended flow:
  - Fork → branch (feat/ or fix/) → tests & docs → PR
- Keep changes small and include unit/integration tests for new functionality.

Need help?
- I can:
  1. Create a PR replacing README.md with this refined version.
  2. Extract exact function signatures & usage examples from app/rag/* and app/ingestion/* to add runnable snippets.
  3. Add a minimal GitHub Actions workflow that runs `python -m pytest` or `python test_phase1.py`.

Which of those would you like me to do next?
