"""BM25 keyword index over the chunk corpus (bm25s, persisted to disk).

RFC prose is dense with exact-match jargon ("SETTINGS_MAX_CONCURRENT_STREAMS",
"Retry-After") that embedding models blur; BM25 keeps those queries sharp.
"""

import json
import logging
from pathlib import Path

import bm25s

from specsage.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, chunks: list[Chunk]):
        self._retriever = retriever
        self._chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        corpus_tokens = bm25s.tokenize([c.embed_text for c in chunks], show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        return cls(retriever, chunks)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(path / "bm25"))
        (path / "chunks.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in self._chunks)
        )
        logger.info("saved BM25 index (%d chunks) to %s", len(self._chunks), path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        retriever = bm25s.BM25.load(str(path / "bm25"))
        chunks = [
            Chunk.model_validate(json.loads(line))
            for line in (path / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return cls(retriever, chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        tokens = bm25s.tokenize([query], show_progress=False)
        limit = min(top_k, len(self._chunks))
        ids, scores = self._retriever.retrieve(tokens, k=limit, show_progress=False)
        return [
            ScoredChunk(chunk=self._chunks[int(i)], score=float(s), source="bm25")
            for i, s in zip(ids[0], scores[0], strict=True)
            if float(s) > 0.0
        ]
