"""Eval runner: retrieval quality per stage, generation quality end-to-end.

Retrieval eval needs Qdrant + the local models. Generation eval additionally
needs the configured LLM (Ollama by default). Both write their raw results to
``evals/results/results.json`` and a human-readable ``report.html``; every
metric quoted in the README comes from these files.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from specsage.agent.pipeline import RagPipeline
from specsage.config import get_settings
from specsage.evals import metrics
from specsage.evals.dataset import EvalQuestion, load_dataset, validate_labels
from specsage.evals.groundedness import score_answer
from specsage.evals.report import render_report
from specsage.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evals/results")
STAGES = ("vector", "bm25", "fused", "reranked")
K = 5


def _eval_retrieval(retriever: HybridRetriever, questions: list[EvalQuestion]) -> dict[str, Any]:
    per_stage: dict[str, dict[str, list[float]]] = {
        s: {"p": [], "r": [], "rr": []} for s in STAGES
    }
    per_question = []
    for q in questions:
        stages = retriever.retrieve_stages(q.question)
        row: dict[str, Any] = {"id": q.id, "question": q.question}
        for stage in STAGES:
            chunks = [sc.chunk for sc in getattr(stages, stage)]
            p = metrics.precision_at_k(chunks, q.labels, K)
            r = metrics.recall_at_k(chunks, q.labels, K)
            rr = metrics.reciprocal_rank(chunks, q.labels)
            per_stage[stage]["p"].append(p)
            per_stage[stage]["r"].append(r)
            per_stage[stage]["rr"].append(rr)
            row[stage] = {"p@5": round(p, 3), "r@5": round(r, 3), "rr": round(rr, 3)}
        per_question.append(row)
        logger.info("retrieval %s: reranked rr=%.2f", q.id, per_stage["reranked"]["rr"][-1])

    summary = {
        stage: {
            f"precision@{K}": round(metrics.aggregate(v["p"]), 3),
            f"recall@{K}": round(metrics.aggregate(v["r"]), 3),
            "mrr": round(metrics.aggregate(v["rr"]), 3),
        }
        for stage, v in per_stage.items()
    }
    return {"summary": summary, "per_question": per_question}


async def _eval_generation(
    pipeline: RagPipeline, questions: list[EvalQuestion]
) -> dict[str, Any]:
    from specsage.retrieval.embedder import embed_texts

    per_question = []
    refusal = {"in_answered": 0, "in_refused": 0, "out_refused": 0, "out_answered": 0}
    total_valid_markers = 0
    total_invalid_markers = 0
    supported_segments = 0
    total_segments = 0

    for q in questions:
        start = time.perf_counter()
        result = await pipeline.ask(q.question)
        elapsed = round(time.perf_counter() - start, 1)

        if q.scope == "in":
            refusal["in_refused" if result.refused else "in_answered"] += 1
        else:
            refusal["out_refused" if result.refused else "out_answered"] += 1

        row: dict[str, Any] = {
            "id": q.id,
            "scope": q.scope,
            "question": q.question,
            "refused": result.refused,
            "seconds": elapsed,
            "answer": result.answer,
        }

        if not result.refused:
            total_valid_markers += len(result.citations)
            total_invalid_markers += len(result.invalid_markers)
            context_texts = {
                i: sc.chunk.text for i, sc in enumerate(result.context, start=1)
            }
            scores = score_answer(result.answer, context_texts, embed_texts)
            supported = sum(1 for s in scores if s.supported)
            supported_segments += supported
            total_segments += len(scores)
            row.update(
                {
                    "citations": len(result.citations),
                    "invalid_markers": len(result.invalid_markers),
                    "segments": len(scores),
                    "segments_supported": supported,
                    "segment_detail": [
                        {
                            "text": s.text[:160],
                            "markers": s.markers,
                            "overlap": round(s.overlap, 3),
                            "cosine": round(s.cosine, 3),
                            "supported": s.supported,
                        }
                        for s in scores
                    ],
                }
            )
        per_question.append(row)
        logger.info("generation %s: refused=%s (%.1fs)", q.id, result.refused, elapsed)

    n_in = refusal["in_answered"] + refusal["in_refused"]
    n_out = refusal["out_refused"] + refusal["out_answered"]
    all_markers = total_valid_markers + total_invalid_markers
    summary = {
        "in_scope_answer_rate": round(refusal["in_answered"] / n_in, 3) if n_in else None,
        "out_of_scope_refusal_rate": (
            round(refusal["out_refused"] / n_out, 3) if n_out else None
        ),
        "citation_validity": (
            round(total_valid_markers / all_markers, 3) if all_markers else None
        ),
        "groundedness": (
            round(supported_segments / total_segments, 3) if total_segments else None
        ),
        "segments_scored": total_segments,
        "refusal_matrix": refusal,
    }
    return {"summary": summary, "per_question": per_question}


def run(generation: bool = True, limit: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    questions = load_dataset()
    retriever = HybridRetriever.from_disk()

    errors = validate_labels(questions, retriever.chunks)
    if errors:
        raise ValueError("dataset labels do not match corpus:\n" + "\n".join(errors))

    if limit:
        in_q = [q for q in questions if q.scope == "in"][:limit]
        out_q = [q for q in questions if q.scope == "out"][: max(2, limit // 4)]
    else:
        in_q = [q for q in questions if q.scope == "in"]
        out_q = [q for q in questions if q.scope == "out"]

    results: dict[str, Any] = {
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "llm_provider": settings.llm_provider,
            "llm_model": (
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else settings.anthropic_model
            ),
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "scope_threshold": settings.scope_threshold,
            "questions_in_scope": len(in_q),
            "questions_out_of_scope": len(out_q),
        },
        "retrieval": _eval_retrieval(retriever, in_q),
    }
    if generation:
        results["generation"] = asyncio.run(_eval_generation(
            RagPipeline.from_settings(), in_q + out_q
        ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "report.html").write_text(render_report(results))
    logger.info("wrote %s and report.html", RESULTS_DIR / "results.json")
    return results
