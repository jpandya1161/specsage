# specsage — Design

**Goal:** a production-shaped RAG system that answers questions over the IETF RFC corpus with
grounded, cited answers — and, unlike most RAG demos, *measures itself*: retrieval quality,
citation validity, and answer groundedness are all computed by a committed evaluation harness.

## Why RFCs

~400 IETF RFCs (HTTP, TLS, QUIC, DNS, WebSocket, OAuth, email families). They are real,
freely redistributable, densely cross-referenced technical documents with a stable section
structure — ideal for section-level citations and multi-hop questions
("How does HTTP/3 flow control differ from HTTP/2's?").

The corpus is **fetched, not committed**: `specsage fetch` downloads the plain-text RFCs from
rfc-editor.org into `data/rfcs/` using a pinned manifest, so the repo stays small and the
pipeline is reproducible.

## Architecture

```
                       ┌────────────────────────────────────────────┐
 question ──► FastAPI ─► agent pipeline                             │
                       │  1. guard: scope check (retrieval score)   │
                       │  2. rewrite: query rewriting/decomposition │
                       │  3. retrieve: hybrid                       │
                       │       vectors (Qdrant, BGE-small)          │
                       │       + keywords (BM25)                    │
                       │       → reciprocal-rank fusion             │
                       │  4. rerank: ONNX cross-encoder             │
                       │  5. generate: LLM w/ [n] citation contract │
                       │  6. verify: citations resolved to sources  │
                       └────────────────┬───────────────────────────┘
                                        │ SSE stream (tokens + source events)
                                        ▼
                                   minimal chat UI
```

### Components

| Component | Choice | Why |
|---|---|---|
| Chunking | section-aware parser for RFC plain-text format | citations point at real section boundaries, not arbitrary windows |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX) | local, free, no torch dependency, deterministic |
| Vector store | Qdrant (docker-compose) | production-grade store, runs locally in one command |
| Keyword search | bm25s | RFCs are jargon-heavy; exact-term match complements vectors |
| Fusion | reciprocal-rank fusion (k=60) | simple, robust, no score-calibration issues |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed | biggest single retrieval-quality lever, still local |
| LLM | provider-agnostic client: **Ollama (default)** or Anthropic | zero-key fresh-clone experience; provider swap is one env var |
| API | FastAPI + SSE | streaming tokens and source events |
| UI | single-page vanilla JS/CSS chat | clean, no framework weight |

### Citation contract

The LLM is prompted to cite sources inline as `[n]` where `n` indexes the retrieved context
blocks. Post-processing maps each marker back to `{rfc, section, title, url}`. A response
whose markers reference non-existent blocks fails verification and the offending markers are
stripped; the eval harness counts this as a citation error.

### Guardrail

If the best fused-and-reranked retrieval score falls below a threshold (tuned on the eval
set), the pipeline refuses with an explicit out-of-scope message instead of letting the LLM
free-associate. The eval set includes out-of-scope questions to measure refusal accuracy.

## Evaluation harness (the point of this project)

`evals/dataset/qa.jsonl`: ~50 curated questions with labeled relevant RFC sections, plus
out-of-scope questions. Labels are validated programmatically (each labeled section must
exist in the chunked corpus).

Metrics, computed by `specsage eval` and rendered into `evals/results/report.html`:

- **Retrieval:** precision@k, recall@k, MRR — evaluated at each stage (BM25 only, vectors
  only, fused, reranked) so the value of each stage is visible.
- **Citation validity:** share of citation markers that resolve to a real retrieved block.
- **Groundedness:** every cited answer sentence is scored for support by its cited chunk
  using embedding similarity + token overlap (deterministic), with an optional LLM-judge
  second opinion. We deliberately do **not** rely on the LLM-judge alone: the default local
  model (8B) is not a trustworthy judge, and the README says so.
- **Refusal accuracy:** out-of-scope questions must be refused; in-scope must not be.

Every number in the README comes from this harness. No invented metrics.

## Testing

- Unit: chunker (structure, edge cases), RRF fusion (ranking math), citation
  parsing/verification, guardrail thresholding.
- Integration: FastAPI endpoints with a fake LLM and an in-memory store (no network, no
  Ollama needed in CI).
- CI (GitHub Actions): ruff + mypy + pytest on every push/PR.

## Non-goals (v1)

- Conversation memory / multi-turn context.
- PDF or HTML ingestion (plain-text RFCs only).
- Cloud deployment (docker-compose local is the deliverable; deployment lands in later
  portfolio projects).
