# TKP — Teacher Knowledge Package (TKP)

A modular Python toolkit for building retrieval-augmented generation (RAG) pipelines tailored to educational content: document ingestion & chunking, hybrid indexing (BM25 + vector), reranking, and grounded LLM generation with citations.

Badges
- Python 3.9+
- Core libs: FastAPI, Streamlit (demo), ChromaDB (vector store)

Quick highlights
- Document parsing for PDFs, PPTX, DOCX, and text with routing for scanned vs text documents.
- Hybrid retrieval: BM25 + vector embeddings, with reranking before LLM generation.
- Grounded, citation-aware generation using an OpenRouter/OpenAI-compatible client.
- Generation orchestration via a graph + node/state pattern for multi-step generation flows.
- Local demo UI and an MLflow-backed evaluation harness for automated scoring.

---

## Table of contents
- What this is
- Quick start
- MLflow evaluation (how it works)
- Project layout
- How to run the demo & evaluation
- Configuration
- Development notes
- Contributing

## What this is
TKP helps convert curriculum documents into retrieval-ready chunks, index them with sparse and dense methods, and run generation workflows that produce teacher-facing lesson plans and supporting materials. It's aimed at researchers and engineers building RAG pipelines for educational content, and includes an evaluation harness that logs runs to MLflow.

## Quick start (short path)
1. Clone and create a venv
```bash
git clone https://github.com/mudgalma/TKP.git
cd TKP
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

2. Install the project
```bash
pip install -e .
```

3. Configure
```bash
cp .env.example .env
# Edit .env and populate keys (OPENROUTER_API_KEY is required)
```

4. Run the smoke test
```bash
python test_phase1.py
```

5. Run the demo UI (optional)
- FastAPI backend (if available):
```bash
uvicorn app:app --reload --port 8000
```
- Streamlit demo:
```bash
streamlit run frontend/main.py
```

---

## MLflow evaluation — how it works
TKP includes an evaluation harness (scripts/run_tkp_eval.py) that:

1. Loads ground-truth JSON scenarios from `ground_truth/` and matches them to source PDFs in `docs/`.
2. Runs an end-to-end pipeline for each scenario (parse → chunk → embed → index → generate) via a `run_pipeline` wrapper.
3. Uses a set of deterministic scorers (Python functions decorated with `@mlflow.genai.scorer`) plus an LLM-as-a-judge Guidelines scorer to evaluate outputs.
4. Calls `mlflow.genai.evaluate(...)` to run and log the evaluation. Each run is recorded by MLflow with metrics, tags, and artifacts (the generated package and any diagnostics).

Key implementation points
- The evaluation script dynamically builds a dataset from `ground_truth/*.json` and will add adversarial test cases (irrelevant queries, tampered documents) when applicable.
- Scorers implemented in `scripts/run_tkp_eval.py` include checks for schema validity, populated teacher scripts, correct handling of adversarial queries, and detection of tampered facts.
- The predict function (`run_pipeline`) executes the same ingestion, retrieval, and generation pipeline used in production and returns a serializable dict (or Pydantic model dump) that MLflow logs.

How to run an evaluation and view results
1. Start the MLflow UI (from the repo root):
```bash
# Starts a local MLflow UI serving the default ./mlruns directory
mlflow ui --backend-store-uri ./mlruns --port 5000
```
2. In another terminal, run the evaluation script:
```bash
python scripts/run_tkp_eval.py
```
3. Open the UI at http://127.0.0.1:5000 to inspect experiments, runs, traces, and metrics.

You should see experiment dashboards like the screenshots below (place the images in `assets/` with the filenames shown to render them here):

![MLflow Overview](assets/mlflow-overview.png)

![MLflow Evaluation runs](assets/mlflow-eval-runs.png)

Notes
- The repository's evaluation code uses `mlflow.genai` helpers (Guidelines scorer and `mlflow.genai.evaluate`) to simplify LLM evaluation workflows.
- The script prints a convenience message with how to start the MLflow server. If you prefer a different backend store (S3 / remote DB), point `--backend-store-uri` to that location.

---

## Project layout (top-level)
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
ground_truth/           # evaluation scenarios (JSON) used by the MLflow harness
scripts/                # developer utilities (including scripts/run_tkp_eval.py)
test_phase1.py          # experiment / smoke test runner
pyproject.toml          # project metadata & declared dependencies
.env.example            # example env variables
initial_tkp.pdf         # project design brief / notes
```

How it fits together
- Parsing & chunking: `app/ingestion` normalizes documents into blocks and splits them into retrieval-sized chunks.
- Indexing & retrieval: `app/rag` contains BM25 (sparse) and vector (dense) paths, plus reranking logic.
- Generation: `app/generation` implements a graph-based orchestration for multi-step generation and produces a Teacher Knowledge Package (TKP) Pydantic model.
- Evaluation: `scripts/run_tkp_eval.py` wraps the pipeline and submits runs to MLflow; scorers live alongside this script.

---

## Configuration
See `app/config.py` for exact env names and defaults. Important env vars:
- OPENROUTER_API_KEY (required)
- LLAMA_CLOUD_API_KEY (optional)
- EMBEDDING_MODEL, RERANKER_MODEL
- CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

---

## Development notes
- The repo currently lacks unit tests beyond `test_phase1.py`. Add `tests/` and a CI workflow for PRs.
- `pyproject.toml` lists the main runtime dependencies. MLflow and cohere are optional integrations — the evaluation script references `mlflow` and `mlflow.genai`. If you plan to run evaluations, ensure `mlflow` (and `mlflow-genai` if needed) are installed in your environment.

## Contributing
- Fork → branch (feat/ or fix/) → tests & docs → PR
- Keep changes small and include unit/integration tests where appropriate.

---

If you'd like, I can now:
1. Add the two MLflow screenshots to `assets/` (you can upload them or I can add them if you provide the image files or base64).
2. Create a PR replacing `README.md` with this version (done in this commit).
3. Add a minimal GitHub Actions workflow to run `python -m pytest` or `python test_phase1.py` on pushes/PRs.

Tell me which of those to do next, or I'll proceed to add the screenshots if you upload them.
