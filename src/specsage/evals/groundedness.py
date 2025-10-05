"""Deterministic groundedness scoring for cited answers.

An answer is decomposed into *claim segments*: spans of text ending at a
citation group ("QUIC streams are bidirectional or unidirectional [1][3]").
Each segment is scored against the chunks it cites with two signals:

- token overlap: fraction of the segment's content words present in the chunk
- embedding cosine: semantic similarity via the same local embedding model

A segment counts as SUPPORTED if overlap >= 0.5, or overlap >= 0.25 with
cosine >= 0.60. The rule is deliberately transparent and reproducible; an LLM
judge can be layered on top, but with a local 8B model as the only judge we
treat these deterministic signals as primary. Thresholds are documented in
the README and were sanity-checked on real answers during development.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_MARKER_GROUP_RE = re.compile(r"((?:\[\d{1,2}\])+)")
_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "into", "is", "it", "its", "may", "must", "not", "of", "on", "or",
    "shall", "should", "such", "that", "the", "their", "there", "these", "this",
    "to", "was", "were", "which", "will", "with", "would", "when", "can", "each",
    "other", "only", "same", "used", "using", "does", "do",
])

OVERLAP_STRONG = 0.5
OVERLAP_WEAK = 0.25
COSINE_MIN = 0.60


@dataclass
class Segment:
    text: str
    markers: list[int]


@dataclass
class SegmentScore:
    text: str
    markers: list[int]
    overlap: float
    cosine: float

    @property
    def supported(self) -> bool:
        return self.overlap >= OVERLAP_STRONG or (
            self.overlap >= OVERLAP_WEAK and self.cosine >= COSINE_MIN
        )


def split_claim_segments(answer: str, min_chars: int = 20) -> list[Segment]:
    """Split an answer at citation groups; each segment carries its markers."""
    segments: list[Segment] = []
    last = 0
    for m in _MARKER_GROUP_RE.finditer(answer):
        text = answer[last : m.start()].strip()
        markers = [int(x) for x in _MARKER_RE.findall(m.group(1))]
        if len(text) >= min_chars:
            segments.append(Segment(text=text, markers=markers))
        last = m.end()
    return segments


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def token_overlap(segment: str, chunk_text: str) -> float:
    seg_words = _content_words(segment)
    if not seg_words:
        return 0.0
    chunk_words = _content_words(chunk_text)
    return len(seg_words & chunk_words) / len(seg_words)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def score_answer(
    answer: str,
    context_texts: dict[int, str],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> list[SegmentScore]:
    """Score every claim segment against the chunks it cites.

    ``context_texts`` maps citation marker -> chunk text. ``embed_fn`` is
    injectable so tests can run without the real model.
    """
    segments = split_claim_segments(answer)
    scored: list[SegmentScore] = []
    for seg in segments:
        cited = [context_texts[m] for m in seg.markers if m in context_texts]
        if not cited:
            scored.append(SegmentScore(seg.text, seg.markers, overlap=0.0, cosine=0.0))
            continue
        overlap = max(token_overlap(seg.text, text) for text in cited)
        vectors = embed_fn([seg.text, *cited])
        cosine = max(_cosine(vectors[0], v) for v in vectors[1:])
        scored.append(SegmentScore(seg.text, seg.markers, overlap=overlap, cosine=cosine))
    return scored
