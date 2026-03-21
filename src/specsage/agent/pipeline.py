"""The agentic RAG pipeline: rewrite → retrieve → guard → generate → verify.

Events stream out as they happen so the API can forward them over SSE:
  {"type": "queries", ...}   rewritten search queries
  {"type": "sources", ...}   retrieved context (before generation starts)
  {"type": "token", ...}     answer tokens
  {"type": "final", ...}     verified AskResult (citations resolved, markers cleaned)
"""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from specsage.agent.citations import format_context, verify_citations
from specsage.config import Settings, get_settings
from specsage.llm.client import LLMClient, build_llm
from specsage.models import AskResult, ScoredChunk
from specsage.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = """\
You are specsage, an assistant that answers questions about internet protocol \
standards using ONLY the provided RFC excerpts.

Rules:
- Cite excerpts inline with bracketed numbers like [1] or [2][3], placed \
immediately after the claim they support.
- Every factual claim must carry at least one citation.
- Use only information present in the excerpts. Never invent RFC numbers, \
section numbers, or facts.
- If the excerpts do not contain enough information to answer, say exactly \
that — do not guess.
- Be concise and technical. No preamble."""

REWRITE_SYSTEM = """\
You turn a user question about internet protocols into search queries for \
retrieving relevant IETF RFC sections.

Reply with 1 to 3 queries, one per line, nothing else. Each query should be \
short and keyword-dense. For comparison questions, emit one query per thing \
being compared."""

REFUSAL_TEXT = (
    "I can't answer that from the RFC corpus I have indexed. My knowledge covers "
    "IETF standards (HTTP, TLS, QUIC, DNS, email, OAuth, transport and routing "
    "protocols, and related formats) — please ask something in that scope."
)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings | None = None,
        guard_scorer: Callable[[str, list[str]], list[float]] | None = None,
    ):
        """``guard_scorer(question, texts) -> scores`` re-scores the gathered
        context against the ORIGINAL question for the out-of-scope decision.

        Retrieval happens with LLM-rewritten queries, and the rewriter will
        happily turn any question into protocol-sounding search terms — so
        scores against rewritten queries systematically overstate relevance
        for off-topic questions. When no scorer is given, retrieval scores
        are used as-is (weaker guard; tests use this or inject a fake).
        """
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or get_settings()
        self._guard_scorer = guard_scorer

    @classmethod
    def from_settings(cls) -> "RagPipeline":
        from specsage.retrieval.embedder import rerank
        from specsage.retrieval.retriever import HybridRetriever

        settings = get_settings()
        return cls(HybridRetriever.from_disk(), build_llm(settings), settings, guard_scorer=rerank)

    async def _rewrite(self, question: str) -> list[str]:
        try:
            raw = await self._llm.complete(REWRITE_SYSTEM, question)
        except Exception:
            logger.warning("query rewrite failed, using raw question", exc_info=True)
            return [question]
        queries = [q.strip("-• \t") for q in raw.strip().splitlines() if q.strip()]
        queries = [q for q in queries if 3 <= len(q) <= 200][:3]
        return queries or [question]

    def _guard_rescore(
        self, question: str, context: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """Re-score context against the original question for the scope decision."""
        if not context or self._guard_scorer is None:
            return context
        scores = self._guard_scorer(question, [sc.chunk.embed_text for sc in context])
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=score, source="guard")
            for sc, score in zip(context, scores, strict=True)
        ]
        return sorted(rescored, key=lambda s: s.score, reverse=True)

    def _gather_context(self, queries: list[str]) -> list[ScoredChunk]:
        """Retrieve per query and merge by best score per chunk."""
        best: dict[str, ScoredChunk] = {}
        for query in queries:
            for sc in self._retriever.retrieve(query):
                prev = best.get(sc.chunk.id)
                if prev is None or sc.score > prev.score:
                    best[sc.chunk.id] = sc
        merged = sorted(best.values(), key=lambda s: s.score, reverse=True)
        return merged[: self._settings.rerank_top_k]

    async def ask_stream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        queries = await self._rewrite(question)
        yield {"type": "queries", "queries": queries}

        context = self._guard_rescore(question, self._gather_context(queries))
        in_scope = bool(context) and context[0].score >= self._settings.scope_threshold
        if not in_scope and question not in queries:
            # The rewriter sometimes produces queries that retrieve chunks
            # irrelevant to the original question; before refusing, retry
            # retrieval with the raw question itself.
            fallback = self._guard_rescore(question, self._gather_context([question]))
            if fallback and (not context or fallback[0].score > context[0].score):
                context = fallback
                in_scope = context[0].score >= self._settings.scope_threshold
        if not in_scope:
            yield {
                "type": "final",
                "result": AskResult(
                    answer=REFUSAL_TEXT, citations=[], refused=True,
                    rewritten_queries=queries,
                ).model_dump(),
            }
            return

        yield {
            "type": "sources",
            "sources": [
                {
                    "n": i,
                    "label": sc.chunk.label,
                    "title": sc.chunk.section_title,
                    "url": sc.chunk.url,
                    "score": round(sc.score, 4),
                }
                for i, sc in enumerate(context, start=1)
            ],
        }

        prompt = f"Question: {question}\n\nExcerpts:\n{format_context(context)}"
        parts: list[str] = []
        async for token in self._llm.stream(ANSWER_SYSTEM, prompt):
            parts.append(token)
            yield {"type": "token", "text": token}

        cleaned, citations, invalid = verify_citations("".join(parts), context)
        if invalid:
            logger.warning("stripped invalid citation markers: %s", invalid)
        yield {
            "type": "final",
            "result": AskResult(
                answer=cleaned,
                citations=citations,
                refused=False,
                rewritten_queries=queries,
                context=context,
                invalid_markers=invalid,
            ).model_dump(),
        }

    async def ask(self, question: str) -> AskResult:
        """Non-streaming convenience wrapper (CLI, evals)."""
        result: AskResult | None = None
        async for event in self.ask_stream(question):
            if event["type"] == "final":
                result = AskResult.model_validate(event["result"])
        assert result is not None
        return result
