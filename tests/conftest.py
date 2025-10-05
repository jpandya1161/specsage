"""Shared fakes: an in-memory retriever and a scripted LLM.

They let pipeline and API tests run with no network, no Qdrant, no Ollama.
"""

from collections.abc import AsyncIterator

import pytest

from specsage.models import Chunk, ScoredChunk


def make_chunk(cid: str, rfc: int = 9110, section: str = "6.1", text: str = "") -> Chunk:
    return Chunk(
        id=cid,
        rfc=rfc,
        rfc_title="HTTP Semantics",
        section=section,
        section_title="Status Codes",
        position=0,
        text=text or f"The content of {cid}. Status codes are three-digit integers.",
        url=f"https://www.rfc-editor.org/rfc/rfc{rfc}.html#section-{section}",
    )


class FakeRetriever:
    def __init__(self, results: list[ScoredChunk]):
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.results


class FakeLLM:
    """Returns a fixed rewrite and a fixed answer; streams the answer in pieces."""

    def __init__(self, rewrite: str = "status codes", answer: str = "Answer. [1]"):
        self.rewrite = rewrite
        self.answer = answer
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, system: str, prompt: str) -> str:
        self.prompts.append((system, prompt))
        return self.rewrite

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        self.prompts.append((system, prompt))
        for i in range(0, len(self.answer), 7):
            yield self.answer[i : i + 7]


@pytest.fixture
def context_chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk("rfc9110-s6.1-0"), score=5.0, source="reranked"),
        ScoredChunk(
            chunk=make_chunk("rfc9110-s15.4-0", section="15.4"), score=3.0, source="reranked"
        ),
    ]
