"""Provider-agnostic LLM client over httpx.

Two implementations: Ollama (default — free, local, no API key) and
Anthropic. Swapping providers is a single env var; nothing else in the
pipeline knows which one is running.
"""

import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from specsage.config import Settings


class LLMClient(Protocol):
    async def complete(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> AsyncIterator[str]: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def _body(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        }

    async def complete(self, system: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat", json=self._body(system, prompt, stream=False)
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream(
                "POST", f"{self._base_url}/api/chat", json=self._body(system, prompt, stream=True)
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if content := data.get("message", {}).get("content"):
                    yield content
                if data.get("done"):
                    break


class AnthropicClient:
    def __init__(self, api_key: str, model: str, timeout: float = 120.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _body(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self._model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

    async def complete(self, system: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=self._body(system, prompt, stream=False),
            )
            resp.raise_for_status()
            return "".join(
                block["text"] for block in resp.json()["content"] if block["type"] == "text"
            )

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=self._body(system, prompt, stream=True),
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line.removeprefix("data:").strip())
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta["text"]


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("SPECSAGE_ANTHROPIC_API_KEY is required for provider=anthropic")
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    return OllamaClient(settings.ollama_base_url, settings.ollama_model)
