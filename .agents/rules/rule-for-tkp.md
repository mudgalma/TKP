---
trigger: always_on
---

# Production Standards — Teacher AI Platform (RAG)

Applies to all code generated in this workspace. Do not skip these to move faster —
if a shortcut is taken, comment `# TODO(scope): <reason>` so it's traceable, not silent.

## 1. Before writing any code
- State the plan in 3-5 bullets before generating a new module: inputs, outputs, failure
  modes, where it fits in the pipeline (ingestion vs retrieval vs generation).
- Never invent a library, endpoint, or config key. If unsure it exists, say so and pick
  a known one.
- Keep each file single-responsibility: one file = one job (parser, chunker, retriever,
  reranker, generator — not one giant `rag.py`).

## 2. Error handling (non-negotiable)
- No bare `except:` and no `except Exception: pass`. Every catch must either:
  (a) handle it meaningfully, (b) re-raise as a typed app exception, or (c) log with
  context and return a safe fallback.
- Define typed exceptions per layer, e.g. `ParsingError`, `EmbeddingError`,
  `RetrievalError`, `GenerationError` — all subclassing a base `AppError`. Never let raw
  third-party exceptions (openai.*, fitz.*, chromadb.*) leak past the module boundary.
- Every external call (LLM API, embeddings API, LlamaParse, vector store) must have:
  - a timeout,
  - retry with exponential backoff (max 3 attempts) for transient errors (429, 5xx,
    connection errors) — no retry on 4xx client errors except 429,
  - a clear failure path back to the caller (don't swallow and return empty silently —
    surface `status: "failed"` with a reason in the task/response object).
- File upload endpoint must validate: file type allow-list, max file size, and reject
  empty/corrupt files with a 4xx before any processing starts.
- Any function touching user input (query text, filenames, uploaded content) must
  validate/sanitize before use — no string-formatting raw user input directly into
  prompts, SQL, or file paths.

## 3. Guardrails (RAG-specific)
- **Grounding**: the generation prompt must explicitly instruct the model to answer
  only from the provided context and to say "I don't have enough information in this
  document" when the retrieved chunks don't support an answer — never let it fall back
  to general knowledge silently.
- **Citation integrity**: every claim in a generated answer must map to a chunk actually
  returned by retrieval for that query. Don't let the generator invent a page/section
  citation — pass citation metadata in and require the model to only reference IDs it
  was given.
- **Injection resistance**: treat retrieved document content as data, not instructions.
  Wrap retrieved chunks in clearly delimited blocks in the prompt (e.g. `<context>...
  </context>`) and instruct the model to ignore any instructions found inside that block.
- **Empty/low-confidence retrieval**: if reranker top score is below a defined threshold,
  return a "not found in document" response instead of forcing an answer.
- **Cost/latency guardrails**: cap context sent to the LLM to a max token budget; cap
  number of chunks reranked (e.g. top 30-50 candidates, not the whole corpus); log token
  usage per request.
- **Input limits**: cap max upload size and max question length; reject before they hit
  the embedding/LLM layer.

## 4. Code quality
- Type hints on every function signature (params + return). No `Any` unless genuinely
  dynamic (e.g. raw JSON from an LLM before validation).
- Every public function/class gets a one-line docstring: what it does, not how.
- Config (API keys, model names, thresholds, chunk size, top-k values) lives in one
  `config.py` / `.env` — never hardcoded inline in logic files.
- No secrets in code or logs, ever. Load from environment variables; fail fast at
  startup with a clear message if a required key is missing.
- Prefer small, testable pure functions over deeply nested logic. Max ~40 lines per
  function as a soft ceiling — if longer, extract a helper.
- Consistent naming: `snake_case` for functions/vars, `PascalCase` for classes, files
  named after what they contain (`bm25_index.py`, not `utils2.py`).
- No dead code, no commented-out blocks left in — delete or explain why it's kept.

## 5. Logging & observability (lightweight, not skipped)
- Structured logging (JSON or key=value), not bare `print()`.
- Log at module boundaries: ingestion start/end per stage, retrieval query + result
  count + latency, generation call + token count + latency, all errors with stack trace
  and request/task ID.
- Every long-running task (ingestion) gets a task ID that shows up in every log line
  related to it, so a failure can be traced end to end.

## 6. Testing (minimum bar, even under time pressure)
- Every new module ships with at least one happy-path test and one failure-path test
  (e.g. malformed PDF, empty retrieval result, LLM timeout).
- Mock external calls (LLM, embeddings, LlamaParse) in tests — never hit real APIs in
  the test suite.
- Do not mark a stage "done" without running it against at least one real sample PDF
  end-to-end, not just unit tests in isolation.

## 7. API design
- Every endpoint returns a consistent response envelope: `{ "success": bool, "data":
  ..., "error": { "code": ..., "message": ... } | null }`.
- Validate all request bodies with Pydantic models — no raw `dict` request handling.
- Return proper HTTP status codes (400 validation, 404 not found, 422 unprocessable,
  500 only for truly unexpected errors — never for expected failure modes like "no
  answer found").

## 8. Before marking any stage complete
Run this checklist mentally (or literally, as a comment) before saying a feature is
done:
- [ ] Handles the empty/missing-input case
- [ ] Handles the external API failure case
- [ ] Has a timeout
- [ ] Doesn't leak secrets/stack traces to the client response
- [ ] Has a log line for both success and failure
- [ ] Has type hints and a docstring
- [ ] Was actually run once against real input, not just imagined
