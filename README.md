# specsage

**Agentic RAG over the IETF RFC archive - with grounded citations and an evaluation harness that actually measures them.**

Ask questions about HTTP, TLS, QUIC, DNS, OAuth, email, TCP/IP and more. Answers stream in with inline [n] citations that resolve to exact RFC sections - and every quality claim in this README is produced by a committed, re-runnable eval (specsage eval), not vibes.

> Most portfolio RAG projects ship a retriever, an LLM call, and zero measurement. The point of this one is the measurement: per-stage retrieval metrics, citation-validity checking, deterministic groundedness scoring, and refusal accuracy on out-of-scope questions.

![specsage answering a QUIC connection-migration question with streamed tokens and section-level citations](docs/img/demo.gif)

*Live demo: the agent rewrites the question into targeted queries, streams the answer, and every [n] chip resolves to a scored, linked RFC section.*

## What it does

- **Hybrid retrieval** over 188 RFCs (15,196 section-anchored chunks): dense vectors (BGE-small via ONNX) in Qdrant + BM25 keyword search, merged with reciprocal-rank fusion, then reordered by a local cross-encoder reranker.
- **Agentic query handling**: an LLM rewrites/decomposes your question into up to 3 targeted search queries (comparison questions retrieve both sides).
- **G