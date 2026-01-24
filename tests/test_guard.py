"""The guard must score context against the ORIGINAL question, not the
rewritten queries — otherwise the rewriter (which converts any question into
protocol-sounding search terms) lets off-topic questions through."""

from specsage.agent.pipeline import RagPipeline
from specsage.config import Settings
from tests.conftest import FakeLLM, FakeRetriever


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


async def test_guard_refuses_when_original_question_scores_low(context_chunks):
    """Retriever returns high-scoring chunks (for the rewritten query), but the
    guard scorer says they're irrelevant to the original question → refuse."""
    pipeline = RagPipeline(
        FakeRetriever(context_chunks),  # scores 5.0 and 3.0 — above threshold
        FakeLLM(rewrite="http status code registry"),
        _settings(scope_threshold=0.1),
        guard_scorer=lambda q, texts: [-9.0] * len(texts),
    )
    result = await pipeline.ask("What is the best biryani recipe?")
    assert result.refused is True


async def test_guard_reorders_context_by_original_question_relevance(context_chunks):
    """Guard scores flip the retrieval order; citations must follow guard order."""
    # context_chunks: rfc9110-s6.1-0 (5.0), rfc9110-s15.4-0 (3.0)
    def scorer(q, texts):
        return [2.0 if "15.4" not in t else 8.0 for t in texts]

    pipeline = RagPipeline(
        FakeRetriever(context_chunks),
        FakeLLM(answer="Answer [1]."),
        _settings(scope_threshold=0.1),
        guard_scorer=scorer,
    )
    result = await pipeline.ask("What are redirection status codes?")
    assert result.refused is False
    assert result.context[0].chunk.section == "15.4"
    assert result.citations[0].section == "15.4"  # [1] now maps to the reordered top chunk


async def test_raw_question_fallback_recovers_from_bad_rewrites(context_chunks):
    """Rewritten queries retrieve junk (guard scores low), but the raw question
    retrieves well — the pipeline must retry with the raw question, not refuse."""

    class QueryAwareRetriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, query):
            self.queries.append(query)
            return context_chunks  # same chunks; guard scorer discriminates

    def scorer(question, texts):
        # Called twice: first for rewritten-query context (return low),
        # then for raw-question context (return high).
        scorer.calls += 1
        return [-5.0] * len(texts) if scorer.calls == 1 else [6.0] * len(texts)

    scorer.calls = 0
    retriever = QueryAwareRetriever()
    pipeline = RagPipeline(
        retriever,
        FakeLLM(rewrite="weird rewritten query", answer="ETags validate caches [1]."),
        _settings(scope_threshold=0.1),
        guard_scorer=scorer,
    )
    result = await pipeline.ask("What is an HTTP ETag and what is it used for?")
    assert result.refused is False
    assert retriever.queries == [
        "weird rewritten query",
        "What is an HTTP ETag and what is it used for?",
    ]


async def test_fallback_still_refuses_true_out_of_scope(context_chunks):
    """Raw-question fallback must not rescue genuinely off-topic questions."""
    pipeline = RagPipeline(
        FakeRetriever(context_chunks),
        FakeLLM(rewrite="protocol sounding query"),
        _settings(scope_threshold=0.1),
        guard_scorer=lambda q, texts: [-9.0] * len(texts),  # low both times
    )
    result = await pipeline.ask("What is the best biryani recipe?")
    assert result.refused is True


async def test_no_guard_scorer_falls_back_to_retrieval_scores(context_chunks):
    pipeline = RagPipeline(
        FakeRetriever(context_chunks), FakeLLM(), _settings(scope_threshold=0.1)
    )
    result = await pipeline.ask("What are status codes?")
    assert result.refused is False
