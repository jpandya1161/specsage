"""Local ONNX embeddings and cross-encoder reranking via fastembed.

Both models run on CPU with no API key and are deterministic, which matters
for the evaluation harness: retrieval metrics are reproducible run-to-run.
"""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from specsage.config import get_settings


@lru_cache
def _embedding_model(name: str) -> TextEmbedding:
    return TextEmbedding(model_name=name)


@lru_cache
def _reranker_model(name: str) -> TextCrossEncoder:
    return TextCrossEncoder(model_name=name)


def embed_texts(texts: list[str]) -> np.ndarray:
    # Single-process on purpose: fastembed's multiprocess mode deadlocks with
    # onnxruntime under macOS spawn semantics. Bulk indexing batches at the
    # call site instead so progress is observable.
    model = _embedding_model(get_settings().embedding_model)
    return np.array(list(model.embed(texts, batch_size=128)))


def embed_query(query: str) -> np.ndarray:
    # BGE models expect a retrieval instruction prefix on queries only.
    model = _embedding_model(get_settings().embedding_model)
    return np.array(list(model.query_embed([query])))[0]


def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder relevance score for each document against the query."""
    model = _reranker_model(get_settings().reranker_model)
    return list(model.rerank(query, documents))
