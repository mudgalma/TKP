# Teacher AI Platform (TKP)

The Teacher Knowledge Package (TKP) Generator is an end-to-end AI platform designed to ingest educational documents (textbooks, research papers, lecture notes) and transform them into structured, queryable knowledge.

It features a dual-retrieval RAG engine with cross-encoder reranking to provide strictly grounded answers, alongside an automated LLM classifier that extracts pedagogical metadata during ingestion.

---

## 🏗 Architecture

### 1. Ingestion Pipeline (`app/ingestion/`)
- **Cost-Aware Parsing:** Routes complex, scanned, or math-heavy PDFs to **LlamaParse** (cloud OCR/Markdown) and text-heavy PDFs to **PyMuPDF** (free, local).
- **Smart Chunking:** Semantically splits text using headers (`MarkdownHeaderSplitter`) or overlapping paragraphs. Every chunk tracks its parent Document ID, Section Title, and exact Page Numbers for accurate citation.
- **Stage 2/3 Classification:** During ingestion, a zero-shot LLM pass (Llama 3.1) extracts educational metadata into structured JSON: Subject, Grade Level, Topic, Objectives, and Key Concepts.

### 2. Core RAG Engine (`app/rag/`)
- **Vector Search (ChromaDB):** Embeds chunks using the local CPU model `BAAI/bge-small-en-v1.5` to find semantic matches.
- **Keyword Search (BM25):** Runs parallel exact-keyword search via `rank_bm25` (persisted to disk).
- **Cross-Encoder Reranking:** Merges candidates and re-scores them using `BAAI/bge-reranker-base`. Chunks that fall below a strict relevance threshold are dropped to prevent hallucination.
- **Grounded Generation:** Sends the top chunks to **OpenRouter (Llama 3.1 8B)**. Prompting strictly enforces answering *only* from context and citing the exact page/section.

### 3. Asynchronous API & Frontend (`app/__init__.py` & `frontend/`)
- **FastAPI Backend:** Exposes `/api/upload` which kicks off `BackgroundTasks` so the API remains responsive during heavy parsing/embedding loads.
- **Streamlit UI:** A 3-page web interface (Upload with live progress polling, Document Metadata view, and Chat interface).

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for lightning-fast dependency management)

### 1. Setup Environment
Clone the repository and configure your API keys:

```bash
cp .env.example .env
```
Edit `.env` and add:
- `OPENROUTER_API_KEY`: Required for Llama 3.1 generation and classification.
- `LLAMA_CLOUD_API_KEY`: Optional, required if you want to route complex PDFs to LlamaParse.

### 2. Install Dependencies
```bash
uv sync
```

### 3. Run the Application
You will need two terminal windows to run the asynchronous backend and the frontend side-by-side.

**Terminal 1: Start FastAPI Backend**
```bash
uv run uvicorn app:app --reload --port 8000
```

**Terminal 2: Start Streamlit Frontend**
```bash
uv run streamlit run frontend/main.py
```

Open `http://localhost:8501` in your browser. Upload a PDF, watch the real-time processing progress, view the extracted pedagogical metadata, and chat with the document using verifiable citations!

---

## 🧪 Running Tests
The core RAG engine is covered by automated tests that mock the external APIs to ensure timeout handling, empty-context behavior, and error propagation work as expected.

```bash
uv run pytest test_phase1.py -v
```

---

## 📋 Implementation Notes
This MVP successfully implements Phase 1 (Core Engine), Phase 2 (Async API & Classifier), and Phase 3 (Streamlit UI). 
- Stages 5-10 (Lesson Plan Generation, Quiz Generation, Grading) from the original product roadmap were deferred for this MVP to focus on building an extremely robust, hallucination-resistant retrieval foundation.
