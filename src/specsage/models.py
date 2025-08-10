"""Core domain models shared across ingestion, retrieval, and the API."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable unit: a slice of an RFC section with full provenance."""

    id: str  # e.g. "rfc9110-s6.1-0"
    rfc: int
    rfc_title: str
    section: str  # "6.1", "A.2", or "abstract"
    section_title: str
    position: int  # chunk index within the section
    text: str
    url: str

    @property
    def label(self) -> str:
        return f"RFC {self.rfc} §{self.section}"

    @property
    def embed_text(self) -> str:
        """Text used for embedding/reranking: prefixed with provenance for context."""
        head = f"RFC {self.rfc} ({self.rfc_title}) §{self.section} {self.section_title}"
        return f"{head}: {self.text}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # which retriever produced it: "vector" | "bm25" | "fused" | "reranked"


class Citation(BaseModel):
    """A resolved [n] marker in a generated answer."""

    marker: int
    chunk_id: str
    rfc: int
    section: str
    section_title: str
    url: str


class AskResult(BaseModel):
    """Non-streaming answer shape (also the final SSE event payload)."""

    answer: str
    citations: list[Citation]
    refused: bool = False
    rewritten_queries: list[str] = Field(default_factory=list)
    context: list[ScoredChunk] = Field(default_factory=list)
    invalid_markers: list[int] = Field(default_factory=list)
