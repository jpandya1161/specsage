"""specsage command-line interface."""

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="specsage", description="Agentic RAG over IETF RFCs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the pinned RFC corpus")
    sub.add_parser("stats", help="chunk the fetched corpus and print statistics")
    sub.add_parser("index", help="embed + index the corpus into Qdrant and BM25")
    p_search = sub.add_parser("search", help="debug: run hybrid retrieval for a query")
    p_search.add_argument("query")
    p_ask = sub.add_parser("ask", help="ask a question end-to-end")
    p_ask.add_argument("question")
    sub.add_parser("serve", help="run the API server")
    p_eval = sub.add_parser("eval", help="run the evaluation harness")
    p_eval.add_argument("--no-generation", action="store_true",
                        help="retrieval metrics only (no LLM needed)")
    p_eval.add_argument("--limit", type=int, default=None,
                        help="cap the number of in-scope questions")

    args = parser.parse_args(argv)

    from specsage.config import get_settings

    settings = get_settings()

    if args.command == "fetch":
        from specsage.ingest.fetch import fetch_corpus

        index = fetch_corpus(settings.rfc_dir)
        print(f"fetched {index['count']} RFCs, {len(index['failures'])} failures")
        if index["failures"]:
            print(f"failed: {index['failures']}", file=sys.stderr)
            return 1
        return 0

    if args.command == "stats":
        from collections import Counter

        from specsage.ingest.corpus import load_corpus_chunks

        chunks = load_corpus_chunks(settings.rfc_dir)
        sizes = [len(c.text) for c in chunks]
        per_rfc = Counter(c.rfc for c in chunks)
        print(f"chunks: {len(chunks)} across {len(per_rfc)} RFCs")
        print(f"chars/chunk: min={min(sizes)} median={sorted(sizes)[len(sizes) // 2]} "
              f"max={max(sizes)}")
        print(f"largest RFCs: {per_rfc.most_common(5)}")
        return 0

    if args.command == "index":
        from specsage.ingest.corpus import load_corpus_chunks
        from specsage.retrieval import vector_store
        from specsage.retrieval.bm25 import BM25Index
        from specsage.retrieval.embedder import embed_texts

        chunks = load_corpus_chunks(settings.rfc_dir)
        total = len(chunks)
        print(f"embedding {total} chunks (local ONNX, single process)...", flush=True)
        batch = 1024
        for start in range(0, total, batch):
            part = chunks[start : start + batch]
            vectors = embed_texts([c.embed_text for c in part])
            if start == 0:
                vector_store.recreate_collection(dim=vectors.shape[1])
            vector_store.upsert_chunks(part, vectors.tolist(), start_id=start)
            print(f"  indexed {min(start + batch, total)}/{total}", flush=True)
        print(f"qdrant: {vector_store.count()} points")
        BM25Index.build(chunks).save(settings.bm25_dir)
        print(f"bm25: saved to {settings.bm25_dir}")
        return 0

    if args.command == "search":
        from specsage.retrieval.retriever import HybridRetriever

        stages = HybridRetriever.from_disk().retrieve_stages(args.query)
        for name in ("vector", "bm25", "fused", "reranked"):
            results = getattr(stages, name)[:5]
            print(f"\n== {name} ==")
            for s in results:
                print(f"  {s.score:8.4f}  {s.chunk.label:<22} {s.chunk.section_title[:50]}")
        return 0

    if args.command == "ask":
        import asyncio

        from specsage.agent.pipeline import RagPipeline

        result = asyncio.run(RagPipeline.from_settings().ask(args.question))
        print(result.answer)
        if result.citations:
            print("\nSources:")
            for c in result.citations:
                print(f"  [{c.marker}] RFC {c.rfc} §{c.section} {c.section_title} — {c.url}")
        return 0

    if args.command == "eval":
        import json as _json

        from specsage.evals.runner import run

        results = run(generation=not args.no_generation, limit=args.limit)
        print(_json.dumps(results["retrieval"]["summary"], indent=2))
        if "generation" in results:
            print(_json.dumps(results["generation"]["summary"], indent=2))
        print("report: evals/results/report.html")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "specsage.api.app:create_app",
            factory=True,
            host=settings.api_host,
            port=settings.api_port,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
