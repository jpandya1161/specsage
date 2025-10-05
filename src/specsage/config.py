"""Application settings, loaded from environment / .env with the SPECSAGE_ prefix."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPECSAGE_", env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "ollama"  # "ollama" | "anthropic"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Retrieval
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rfc_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    fetch_top_k: int = 30  # candidates from each retriever before fusion
    rerank_top_k: int = 8  # context blocks handed to the LLM
    rrf_k: int = 60
    # Minimum guard score (cross-encoder vs the original question) for the
    # best candidate; below this we refuse as out-of-scope. Calibrated on the
    # eval set: in-scope questions bottom out at +3.8, out-of-scope peaks at
    # +0.5 (Linux fork(), which shares systems vocabulary) — 2.0 sits mid-gap.
    scope_threshold: float = 2.0

    # Paths
    data_dir: Path = Path("data")
    bm25_dir: Path = Path("data/bm25_index")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def rfc_dir(self) -> Path:
        return self.data_dir / "rfcs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
