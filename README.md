# Teacher AI Platform (TKP)

The Teacher Knowledge Package (TKP) Generator is an end-to-end AI platform that ingests educational documents (textbooks, papers, lecture notes) and turns them into structured, pedagogical knowledge packages you can query and build lesson assets from. It is focused on strict grounding (minimizing hallucination) and extracting classroom-relevant metadata during ingestion.

### Highlights
- Dual-retrieval RAG engine (semantic vectors + BM25) with cross-encoder reranking to ensure relevant, grounded context.
- Automated LLM-based classifier to extract pedagogical metadata (subject, grade level, objectives) while ingesting documents.
- Asynchronous FastAPI backend plus a Streamlit frontend for upload, metadata inspection, and chat-style querying with verifiable citations.

---

## What this is
A minimal viable product (MVP) for converting educational PDFs into a searchable, LLM-grounded knowledge store that returns strict, citation-backed answers for educators and curriculum designers.

### Stack
- **Language(s):** Python 3.11+
- **Framework / runtime:** FastAPI backend, Streamlit frontend
- **Notable libraries / components:** 
  - Vector search and embedding tooling referenced in design (Chroma/CPU embedding model usage)
  - BM25 keyword search (rank_bm25)
  - Cross-encoder reranking for candidate fusion
  - OpenRouter / Llama 3.x for generation/classification (API-backed)
  - uv (uv package) for dependency & run helpers

## How it's organized
Top-level layout (important entries only):

```
.agents/               (agent config / tooling)
.claude/               (agent config / tooling)
.env.example           Example env variables and keys to run services
pyproject.toml         Python project manifest / dependencies
uv.lock                Lock file used by uv
initial_tkp.pdf        Design / initial product spec (PDF)
README.md              This file
test_phase1.py         Basic test harness for core engine
app/                   Backend application and core logic
  __init__.py
  config.py
  parser.py
  document_registry.py
  exceptions.py
  ingestion/           Ingestion pipeline and chunking/classification
    __init__.py
    parser.py
    chunker.py
    classifier.py
  rag/                 Retrieval & generation utilities
    __init__.py
    bm25_index.py
    embed.py
    generate.py
    reranker.py
    retriever.py
    vector_store.py
frontend/              Streamlit UI and pages
  main.py
  pages/
    1_Upload.py
    2_Documents.py
    3_Ask.py
    4_Generator.py
src/
  tkp/                 Lightweight package markers / packaging support
```

How it fits together:
- The ingestion pipeline (app/ingestion/) parses PDFs, OCRs when necessary, splits documents into semantically meaningful chunks (with page/section metadata), and runs a classifier to extract pedagogical tags.
- Chunks are indexed into both a vector store and a BM25 index (app/rag/). The retriever merges candidates from both indexes, reranks them with a cross-encoder, and returns a compact, high-precision context to the generator.
- The FastAPI app exposes upload and query endpoints and delegates heavy work to background tasks. The Streamlit frontend (frontend/) interacts with the backend for uploads, progress, metadata viewing, and chat-style Q&A.

---

## Architecture (summary)
1. Ingestion (app/ingestion/)
   - Flexible parsing that routes scanned/math-heavy PDFs to OCR/parser and text PDFs to local PDF parsing.
   - Chunking preserves document structure: each chunk records Document ID, section title, and page numbers.
   - LLM-based zero-shot classification extracts subject, grade, topic, objectives, and key concepts into JSON.

2. Retrieval & Generation (app/rag/)
   - Embedding-based semantic search (local/CPU model) + BM25 keyword index.
   - Cross-encoder reranker fuses and re-scores candidate contexts, dropping low-relevance items to reduce hallucination.
   - Grounded generation via an API-backed LLM with prompts that enforce citation-only answers.

3. API & Frontend
   - FastAPI backend provides asynchronous endpoints (uploads spawn background tasks).
   - Streamlit frontend contains upload, document metadata, search, and generation pages.

---

## Quickstart — from clone to running

### Prerequisites
- Python 3.11+
- Recommended: use the uv tool referenced in this repo for dependency management and running commands:
  - uv (https://github.com/astral-sh/uv)

### 1. Get the code and configure environment
```bash
git clone https://github.com/mudgalma/TKP.git
cd TKP
cp .env.example .env
```

Edit `.env` and set the required keys:
- `OPENROUTER_API_KEY` — required for LLM generation & classification (OpenRouter / Llama 3.x)
- `LLAMA_CLOUD_API_KEY` — optional; required if routing OCR/complex parsing to a cloud parsing API
- Any other keys your deployment or tests require (check config.py for additional env var names)

### 2. Install dependencies
If you use uv (project uses uv-managed workflow):
```bash
uv sync
```
Or install using pip / virtual environment if you prefer:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # if you produce a requirements file from pyproject.toml
```

### 3. Run the services
Start the backend (FastAPI + Uvicorn):
```bash
uv run uvicorn app:app --reload --port 8000
```

Start the frontend (Streamlit) in a second terminal:
```bash
uv run streamlit run frontend/main.py
```

- Streamlit default URL: http://localhost:8501
- API backend: http://localhost:8000 (endpoints under /api/*)

### 4. Basic workflow
- Open the Streamlit UI, upload a PDF, watch the progress, review extracted metadata in the Documents page, and use the Ask page to query the document. Answers returned by the generator include context references (page/section).

---

## Running tests
A core test harness exists to validate the RAG engine behavior, using mock responses for external APIs:
```bash
uv run pytest test_phase1.py -v
```
(Or run pytest directly in your virtualenv.)

---

## Configuration & important files
- `.env.example` — copy and populate as `.env` to provide required API keys and runtime settings.
- `pyproject.toml` — Python project manifest (dependencies, build settings).
- `initial_tkp.pdf` — original product/design specification used during development.
- `app/config.py` — central config and environment variable references.
- `app/parser.py` & `app/ingestion/*` — parsing, chunking, and classification logic.
- `app/rag/*` — vector/BM25 indexing, retrieval, reranking, and generation glue.

---

## Implementation notes & roadmap
This repository focuses on Phase 1–3 of the original roadmap:
- Phase 1: Core RAG engine with embeddings, BM25, and reranking.
- Phase 2: Async API and classification metadata extraction.
- Phase 3: Streamlit UI for upload, metadata viewing, and chat.

Planned/Deferred:
- Lesson plan generation, auto-quiz generation, and grading workflows were deferred from the MVP to concentrate on reliable grounding and ingestion quality.

---

## Contributing
Contributions are welcome. Recommended steps:
1. Open an issue describing the bug or feature request.
2. Create a branch for your work and open a PR with tests where appropriate.
3. Keep changes focused and include changelog notes for larger features.

Suggested areas for contribution:
- Add more robust PDF/math OCR integration.
- Add unit/integration tests for ingestion edge cases.
- Add persistent vector store adapters or optional cloud vector DB integrations.
- Improve prompting and citation formats for different curriculum styles.

---

## Security & privacy
- This project may send document content to external LLM providers depending on configuration; never upload sensitive or personal data without reviewing your provider's privacy policy.
- API keys should be stored in environment variables and never committed to the repository.

---

## License
Add your chosen license file (e.g., LICENSE) to the repository. If no license is present, contact the repository owner for clarification before using the code in production.

---

## Try asking
- "How do I add a new embedding model adapter to app/rag/embed.py so it uses a hosted embeddings API?"
- "Where does the ingestion pipeline record the source page numbers for each chunk (which file/property)?"
- "The Streamlit Upload page shows progress — which backend endpoint and background task function are responsible for that progress reporting?"
