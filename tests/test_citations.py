from specsage.agent.citations import extract_markers, format_context, verify_citations
from specsage.models import ScoredChunk
from tests.conftest import make_chunk


def _ctx(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk(f"c{i}", section=f"{i}.1"), score=1.0, source="reranked")
        for i in range(1, n + 1)
    ]


def test_extract_markers_orders_by_first_appearance():
    assert extract_markers("A [2] then [1], again [2][3]") == [2, 1, 3]


def test_verify_keeps_valid_markers_and_resolves_them():
    text = "Status codes are three digits [1]. Redirection is 3xx [2]."
    cleaned, citations, invalid = verify_citations(text, _ctx(2))
    assert cleaned == text
    assert invalid == []
    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].chunk_id == "c1"
    assert citations[0].url.endswith("#section-1.1")


def test_verify_strips_fabricated_markers():
    text = "True claim [1]. Fabricated claim [7]."
    cleaned, citations, invalid = verify_citations(text, _ctx(2))
    assert "[7]" not in cleaned
    assert "[1]" in cleaned
    assert invalid == [7]
    assert [c.marker for c in citations] == [1]


def test_verify_handles_no_markers():
    cleaned, citations, invalid = verify_citations("No citations here.", _ctx(2))
    assert cleaned == "No citations here."
    assert citations == [] and invalid == []


def test_format_context_numbers_match_contract():
    rendered = format_context(_ctx(2))
    assert rendered.index("[1] RFC 9110 §1.1") < rendered.index("[2] RFC 9110 §2.1")
