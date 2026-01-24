import pytest

from specsage.agent.pipeline import REFUSAL_TEXT, RagPipeline
from specsage.config import Settings
from tests.conftest import FakeLLM, FakeRetriever


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


async def test_full_flow_streams_and_verifies(context_chunks):
    llm = FakeLLM(rewrite="status codes\nredirection 3xx", answer="Codes are 3 digits [1][9].")
    pipeline = RagPipeline(FakeRetriever(context_chunks), llm, _settings(scope_threshold=0.1))

    events = [e async for e in pipeline.ask_stream("What are status codes?")]
    types = [e["type"] for e in events]
    assert types[0] == "queries"
    assert "sources" in types and "token" in types
    assert types[-1] == "final"

    final = events[-1]["result"]
    assert final["refused"] is False
    # invalid marker [9] stripped, valid [1] kept and resolved
    assert "[9]" not in final["answer"] and "[1]" in final["answer"]
    assert [c["marker"] for c in final["citations"]] == [1]
    # both rewritten queries were used for retrieval
    assert events[0]["queries"] == ["status codes", "redirection 3xx"]

    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "Codes are 3 digits [1][9]."


async def test_out_of_scope_question_is_refused(context_chunks):
    low = [c.model_copy(update={"score": 0.01}) for c in context_chunks]
    pipeline = RagPipeline(FakeRetriever(low), FakeLLM(), _settings(scope_threshold=0.5))

    result = await pipeline.ask("What is the best biryani recipe?")
    assert result.refused is True
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


async def test_empty_retrieval_is_refused():
    pipeline = RagPipeline(FakeRetriever([]), FakeLLM(), _settings())
    result = await pipeline.ask("anything")
    assert result.refused is True


async def test_rewrite_failure_falls_back_to_raw_question(context_chunks):
    class BrokenRewriteLLM(FakeLLM):
        async def complete(self, system: str, prompt: str) -> str:
            raise RuntimeError("ollama down")

    retriever = FakeRetriever(context_chunks)
    pipeline = RagPipeline(retriever, BrokenRewriteLLM(), _settings())
    result = await pipeline.ask("What are status codes?")
    assert result.rewritten_queries == ["What are status codes?"]
    assert retriever.queries == ["What are status codes?"]


async def test_context_capped_at_rerank_top_k(context_chunks):
    settings = _settings(rerank_top_k=1)
    pipeline = RagPipeline(FakeRetriever(context_chunks), FakeLLM(), settings)
    result = await pipeline.ask("status codes")
    assert len(result.context) == 1
    assert result.context[0].chunk.id == "rfc9110-s6.1-0"  # highest score wins


@pytest.mark.parametrize("provider", ["ollama", "anthropic"])
def test_build_llm_provider_selection(provider):
    from specsage.llm.client import AnthropicClient, OllamaClient, build_llm

    settings = _settings(llm_provider=provider, anthropic_api_key="k")
    client = build_llm(settings)
    expected = AnthropicClient if provider == "anthropic" else OllamaClient
    assert isinstance(client, expected)
