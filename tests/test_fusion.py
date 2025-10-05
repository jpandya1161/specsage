from specsage.models import Chunk, ScoredChunk
from specsage.retrieval.fusion import reciprocal_rank_fusion


def _sc(cid: str, score: float, source: str) -> ScoredChunk:
    chunk = Chunk(
        id=cid, rfc=1, rfc_title="t", section="1", section_title="s",
        position=0, text="x" * 60, url="https://example.org",
    )
    return ScoredChunk(chunk=chunk, score=score, source=source)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    vec = [_sc("a", 0.9, "vector"), _sc("b", 0.8, "vector"), _sc("c", 0.7, "vector")]
    bm = [_sc("b", 12.0, "bm25"), _sc("d", 9.0, "bm25"), _sc("a", 5.0, "bm25")]
    fused = reciprocal_rank_fusion([vec, bm], k=60)
    order = [s.chunk.id for s in fused]
    # "a" (ranks 1,3) and "b" (ranks 2,1) beat single-list items
    assert set(order[:2]) == {"a", "b"}
    assert order.index("b") < order.index("d")


def test_rrf_scores_are_rank_based_not_score_based():
    # Wildly different score scales must not matter
    vec = [_sc("a", 0.0001, "vector")]
    bm = [_sc("a", 99999.0, "bm25")]
    fused = reciprocal_rank_fusion([vec, bm], k=60)
    assert len(fused) == 1
    assert abs(fused[0].score - 2 / 61) < 1e-9


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    single = [_sc("a", 1.0, "vector")]
    fused = reciprocal_rank_fusion([single, []])
    assert [s.chunk.id for s in fused] == ["a"]


def test_rrf_output_is_sorted_and_deduplicated():
    vec = [_sc("a", 0.9, "vector"), _sc("b", 0.8, "vector")]
    bm = [_sc("a", 3.0, "bm25")]
    fused = reciprocal_rank_fusion([vec, bm])
    ids = [s.chunk.id for s in fused]
    assert ids == ["a", "b"]
    assert all(s.source == "fused" for s in fused)
    scores = [s.score for s in fused]
    assert scores == sorted(scores, reverse=True)
