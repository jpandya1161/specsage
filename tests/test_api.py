import json

from fastapi.testclient import TestClient

from specsage.agent.pipeline import RagPipeline
from specsage.api.app import create_app
from specsage.config import Settings
from tests.conftest import FakeLLM, FakeRetriever


def _client(context, llm=None) -> TestClient:
    pipeline = RagPipeline(
        FakeRetriever(context), llm or FakeLLM(), Settings(_env_file=None, scope_threshold=0.1)
    )
    return TestClient(create_app(pipeline))


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def test_health(context_chunks):
    resp = _client(context_chunks).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_streams_sse_events(context_chunks):
    llm = FakeLLM(answer="Three-digit codes [1].")
    with _client(context_chunks, llm).stream(
        "POST", "/ask", json={"question": "What are HTTP status codes?"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.read().decode()

    events = _sse_events(body)
    types = [e["type"] for e in events]
    assert types[0] == "queries"
    assert "sources" in types
    final = next(e for e in events if e["type"] == "final")
    assert final["result"]["citations"][0]["rfc"] == 9110
    assert body.rstrip().endswith("data: [DONE]")


def test_ask_refuses_out_of_scope():
    with _client([]).stream("POST", "/ask", json={"question": "best biryani recipe?"}) as resp:
        body = resp.read().decode()
    final = next(e for e in _sse_events(body) if e["type"] == "final")
    assert final["result"]["refused"] is True


def test_ask_validates_input(context_chunks):
    resp = _client(context_chunks).post("/ask", json={"question": "hi"})
    assert resp.status_code == 422
    resp = _client(context_chunks).post("/ask", json={})
    assert resp.status_code == 422


def test_pipeline_error_streams_error_event(context_chunks):
    class ExplodingLLM(FakeLLM):
        async def complete(self, system, prompt):  # rewrite ok
            return "q"

        async def stream(self, system, prompt):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    with _client(context_chunks, ExplodingLLM()).stream(
        "POST", "/ask", json={"question": "What are status codes?"}
    ) as resp:
        body = resp.read().decode()
    events = _sse_events(body)
    assert any(e["type"] == "error" for e in events)
