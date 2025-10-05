"""Parsing and verification of the [n] citation contract.

The LLM is instructed to cite context blocks as [1], [2][3], etc. We treat
its output as untrusted: markers pointing at non-existent blocks are stripped
and reported, so a fabricated citation can never reach the user as if real.
"""

import re

from specsage.models import Citation, ScoredChunk

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def extract_markers(text: str) -> list[int]:
    """All citation markers in order of first appearance."""
    seen: list[int] = []
    for m in _MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def verify_citations(
    text: str, context: list[ScoredChunk]
) -> tuple[str, list[Citation], list[int]]:
    """Resolve markers against the context blocks.

    Returns (cleaned_text, valid_citations, invalid_markers). Invalid markers
    are removed from the text.
    """
    valid_markers = set(range(1, len(context) + 1))
    invalid = sorted({int(m) for m in _MARKER_RE.findall(text)} - valid_markers)

    def _replace(m: re.Match[str]) -> str:
        return m.group(0) if int(m.group(1)) in valid_markers else ""

    cleaned = _MARKER_RE.sub(_replace, text)

    citations = []
    for n in extract_markers(cleaned):
        chunk = context[n - 1].chunk
        citations.append(
            Citation(
                marker=n,
                chunk_id=chunk.id,
                rfc=chunk.rfc,
                section=chunk.section,
                section_title=chunk.section_title,
                url=chunk.url,
            )
        )
    return cleaned, citations, invalid


def format_context(context: list[ScoredChunk]) -> str:
    """Render context blocks the way the citation contract numbers them."""
    blocks = []
    for i, sc in enumerate(context, start=1):
        c = sc.chunk
        blocks.append(f"[{i}] RFC {c.rfc} §{c.section} ({c.section_title}):\n{c.text}")
    return "\n\n".join(blocks)
