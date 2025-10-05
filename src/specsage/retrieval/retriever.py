"""Hybrid retrieval orchestration: vectors + BM25 → RRF → cross-encoder rerank.

``SearchStages`` keeps every intermediate ranking so the evaluation harness
can attribute quality to each stage instead of treating retrieval as a black
box.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from specsage.config import get_settings
from specsage.models import ScoredChunk
from specsage.retrieval import vector_store
from specsage.retrieval.bm25 import BM25Index
from specsage.retrieval.embedder import embed_query, rerank
from specsage.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class SearchStages:
    vector: list[ScoredChunk] = field(default_factory=list)
    bm25: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    reranked: list[ScoredChunk] = field(default_factory=list)


class Retriever(Protocol):
    """What the agent pipeline needs; tests substitute an in-memory fake."""

    def retrieve(self, query: str) -> list[ScoredChunk]: ...


class HybridRetriever:
    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index
        self._settings = get_settings()

    @classmethod
    def from_disk(cls) -> "HybridRetriever":
        return cls(BM25Index.load(get_settings().bm25_dir))

    @property
    def chunks(self):
        return self._bm25.chunks

    def retrieve_stages(self, query: str) -> SearchStages:
        s = self._settings
        stages = SearchStages()
        stages.vector = vector_store.search(list(embed_query(query)), s.fetch_top_k)
        stages.bm25 = self._bm25.search(query, s.fetch_top_k)
        stages.fused = reciprocal_rank_fusion([stages.vector, stages.bm25], k=s.rrf_k)

        candidates = stages.fused[: s.fetch_top_k]
        if candidates:
            scores = rerank(query, [c.chunk.embed_text for c in candidates])
            reranked = [
                ScoredChunk(chunk=c.chunk, score=score, source="reranked")
                for c, score in zip(candidates, scores, strict=True)
            ]
            reranked.sort(key=lambda s: s.score, reverse=True)
            stages.reranked = reranked[: s.rerank_top_k]
        return stages

    def retrieve(self, query: str) -> list[ScoredChunk]:
        return self.retrieve_stages(query).reranked
