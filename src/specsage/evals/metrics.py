"""Retrieval metrics: precision@k, recall@k (label coverage), MRR."""

from specsage.evals.dataset import Label, chunk_matches_any, section_matches
from specsage.models import Chunk


def precision_at_k(retrieved: list[Chunk], labels: list[Label], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for c in top if chunk_matches_any(c, labels))
    return hits / len(top)


def recall_at_k(retrieved: list[Chunk], labels: list[Label], k: int) -> float:
    """Fraction of labeled sections covered by at least one retrieved chunk."""
    if not labels:
        return 0.0
    top = retrieved[:k]
    covered = sum(
        1
        for label in labels
        if any(c.rfc == label.rfc and section_matches(c.section, label.section) for c in top)
    )
    return covered / len(labels)


def reciprocal_rank(retrieved: list[Chunk], labels: list[Label]) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk_matches_any(chunk, labels):
            return 1.0 / rank
    return 0.0


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
