"""Tests for eval metrics, dataset validation, and groundedness scoring."""

import numpy as np
import pytest

from specsage.evals.dataset import (
    EvalQuestion,
    Label,
    chunk_matches_any,
    load_dataset,
    section_matches,
    validate_labels,
)
from specsage.evals.groundedness import (
    score_answer,
    split_claim_segments,
    token_overlap,
)
from specsage.evals.metrics import precision_at_k, recall_at_k, reciprocal_rank
from tests.conftest import make_chunk

L = [Label(rfc=9110, section="15.4")]


def test_section_matching_covers_subsections_only():
    assert section_matches("15.4", "15.4")
    assert section_matches("15.4.5", "15.4")
    assert not section_matches("15.40", "15.4")  # not a child, just a prefix string
    assert not section_matches("15", "15.4")


def test_chunk_matching_requires_same_rfc():
    chunk = make_chunk("x", rfc=9112, section="15.4")
    assert not chunk_matches_any(chunk, L)
    assert chunk_matches_any(make_chunk("y", rfc=9110, section="15.4.5"), L)


def test_precision_and_recall_at_k():
    retrieved = [
        make_chunk("a", rfc=9110, section="15.4.5"),  # relevant
        make_chunk("b", rfc=9110, section="6.1"),     # not
        make_chunk("c", rfc=8446, section="2.3"),     # not
    ]
    assert precision_at_k(retrieved, L, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, L, 3) == 1.0
    assert recall_at_k(retrieved, L, 0) == 0.0
    assert reciprocal_rank(retrieved, L) == 1.0
    assert reciprocal_rank(retrieved[1:], L) == 0.0


def test_multi_label_recall_counts_each_label():
    labels = [Label(rfc=1, section="1"), Label(rfc=1, section="2")]
    retrieved = [make_chunk("a", rfc=1, section="1.5")]
    assert recall_at_k(retrieved, labels, 5) == 0.5


def test_shipped_dataset_is_well_formed():
    questions = load_dataset()
    assert len(questions) >= 50
    assert sum(1 for q in questions if q.scope == "out") >= 8
    assert all(q.labels for q in questions if q.scope == "in")


def test_validate_labels_reports_missing_sections():
    qs = [EvalQuestion(id="x", scope="in", question="?", labels=[Label(rfc=42, section="9.9")])]
    errors = validate_labels(qs, [make_chunk("a", rfc=42, section="1")])
    assert errors and "RFC 42" in errors[0]
    assert not validate_labels(qs, [make_chunk("a", rfc=42, section="9.9.1")])


# ── groundedness ──────────────────────────────────────────────


def test_split_claim_segments():
    answer = "Status codes are three digits [1]. They appear in responses [2][3]. Trailing text."
    segments = split_claim_segments(answer)
    assert len(segments) == 2
    assert segments[0].markers == [1]
    assert segments[1].markers == [2, 3]
    assert "Trailing" not in segments[0].text + segments[1].text


def test_token_overlap_ignores_stopwords():
    assert token_overlap("the code is a three-digit integer", "three-digit integer code") == 1.0
    assert token_overlap("quantum biryani", "three-digit integer code") == 0.0


def test_score_answer_supported_and_unsupported():
    def fake_embed(texts):
        # First text is the segment; make it identical to supported chunk vector
        vectors = {"seg": [1.0, 0.0], "same": [1.0, 0.0], "other": [0.0, 1.0]}
        return np.array([vectors["seg"]] + [vectors[t] for t in texts[1:]])

    answer = "The status code is a three-digit integer code [1]. Bananas are yellow fruit [2]."
    context = {1: "same", 2: "other"}

    # patch token overlap inputs: chunk 1 text "same" won't overlap; rely on cosine
    scores = score_answer(answer, context, fake_embed)
    assert len(scores) == 2
    assert scores[0].cosine == pytest.approx(1.0)
    assert scores[1].cosine == pytest.approx(0.0)
    assert not scores[1].supported


def test_score_answer_handles_marker_without_context():
    scores = score_answer("Some long enough claim text here [4].", {}, lambda t: np.zeros((1, 2)))
    assert len(scores) == 1
    assert scores[0].overlap == 0.0 and not scores[0].supported
