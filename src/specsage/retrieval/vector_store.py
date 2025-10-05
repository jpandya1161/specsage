"""Qdrant-backed vector store for RFC chunks."""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from specsage.config import get_settings
from specsage.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_BATCH = 256


def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _chunk_from_payload(payload: dict) -> Chunk:
    return Chunk.model_validate(payload)


def recreate_collection(dim: int) -> None:
    client = _client()
    name = get_settings().collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]], start_id: int = 0) -> None:
    client = _client()
    name = get_settings().collection
    points = [
        PointStruct(id=start_id + i, vector=vec, payload=chunk.model_dump())
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]
    for start in range(0, len(points), _BATCH):
        client.upsert(collection_name=name, points=points[start : start + _BATCH])
    logger.info("upserted %d points into %s", len(points), name)


def search(query_vector: list[float], top_k: int) -> list[ScoredChunk]:
    client = _client()
    hits = client.query_points(
        collection_name=get_settings().collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        ScoredChunk(chunk=_chunk_from_payload(h.payload or {}), score=h.score, source="vector")
        for h in hits
    ]


def count() -> int:
    return _client().count(get_settings().collection).count
