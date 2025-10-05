"""Reciprocal-rank fusion of multiple ranked retrieval lists.

RRF is rank-based, so it needs no score calibration between BM25 (unbounded)
and cosine similarity (bounded) — the standard trick for hybrid retrieval.
"""

from collections import defaultdict

from specsage.models import ScoredChunk


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank_d)."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item.chunk.id
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in best:
                best[cid] = item

    fused = [
        ScoredChunk(chunk=best[cid].chunk, score=score, source="fused")
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda s: s.score, reverse=True)
    return fused
