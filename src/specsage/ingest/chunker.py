"""Section-aware chunking of RFC plain-text documents.

RFCs have a stable visual grammar we exploit instead of blind fixed-size
windows: headings sit at column 0 ("6.1.  Status Codes", "Appendix A.  ..."),
body text is indented, and legacy documents are paginated with form feeds and
running headers/footers. Chunks therefore align with real section boundaries,
which is what makes section-level citations possible.
"""

import re
from dataclasses import dataclass, field

from specsage.models import Chunk

# Legacy pagination: "Fielding, et al.   Standards Track   [Page 5]"
_FOOTER_RE = re.compile(r"^\s*\S.*\[Page \d+\]\s*$")
# Running header after a page break: "RFC 9110    HTTP Semantics    June 2022"
_HEADER_RE = re.compile(r"^RFC \d+\s{2,}.*\s{2,}\S.*$")
# Numbered heading at column 0: "6.1.  Status Codes"
_NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s{1,4}(\S.*)$")
# Appendix heading: "Appendix A.  Collected ABNF" / "A.2.  Changes"
_APPENDIX_HEADING_RE = re.compile(r"^Appendix ([A-Z])(?:\.)?\s{1,4}(\S.*)$")
_APPENDIX_SUB_RE = re.compile(r"^([A-Z](?:\.\d+)+)\.?\s{1,4}(\S.*)$")
# Table-of-contents entries end in dot leaders + a page number
_TOC_TAIL_RE = re.compile(r"(\.\s?){2,}\s*\d+\s*$")

_UNNUMBERED = {
    "abstract",
    "status of this memo",
    "copyright notice",
    "table of contents",
    "introduction",
    "security considerations",
    "iana considerations",
    "acknowledgements",
    "acknowledgments",
    "contributors",
    "authors' addresses",
    "author's address",
    "index",
    "full copyright statement",
    "references",
    "normative references",
    "informative references",
}

# Sections that are boilerplate or citation noise, never worth retrieving.
_SKIP_TITLES = re.compile(
    r"(references|acknowledg|contributors|authors?' address|table of contents"
    r"|status of this memo|copyright|full copyright|intellectual property|^index$)",
    re.IGNORECASE,
)


@dataclass
class Section:
    number: str  # "6.1", "A.2", "abstract"
    title: str
    lines: list[str] = field(default_factory=list)


def normalize(text: str) -> list[str]:
    """Strip legacy pagination artifacts; return clean lines."""
    lines: list[str] = []
    after_break = False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if "\f" in line:
            after_break = True
            line = line.replace("\f", "")
            if not line:
                continue
        if _FOOTER_RE.match(line) and "[Page" in line:
            continue
        if after_break and (_HEADER_RE.match(line) or not line):
            if _HEADER_RE.match(line):
                after_break = False
            continue
        after_break = False
        lines.append(line)
    return lines


def _match_heading(line: str) -> tuple[str, str] | None:
    """Return (section_number, title) if the line is a section heading."""
    if _TOC_TAIL_RE.search(line):
        return None
    if m := _NUM_HEADING_RE.match(line):
        title = m.group(2).strip()
        # Reject sentence-like false positives: headings are short and don't
        # end mid-sentence punctuation like ',' or ';'.
        if len(title) > 90 or title.endswith((",", ";", ":")):
            return None
        return m.group(1), title
    if m := _APPENDIX_HEADING_RE.match(line):
        return m.group(1), m.group(2).strip()
    if m := _APPENDIX_SUB_RE.match(line):
        return m.group(1), m.group(2).strip()
    stripped = line.strip().rstrip(":").lower()
    if line and not line[0].isspace() and stripped in _UNNUMBERED:
        return stripped, line.strip()
    return None


def parse_sections(text: str) -> list[Section]:
    """Split a normalized RFC into sections; content before the first heading is dropped."""
    lines = normalize(text)
    sections: list[Section] = []
    current: Section | None = None
    in_toc = False

    for line in lines:
        heading = _match_heading(line)
        if heading:
            number, title = heading
            in_toc = title.strip().lower() == "table of contents"
            current = Section(number=number, title=title)
            sections.append(current)
            continue
        if current is not None and not in_toc:
            current.lines.append(line)

    return [s for s in sections if not _SKIP_TITLES.search(s.title)]


def _paragraphs(lines: list[str]) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip():
            buf.append(line)
        elif buf:
            paras.append("\n".join(buf))
            buf = []
    if buf:
        paras.append("\n".join(buf))
    return paras


def _pack(paras: list[str], max_chars: int) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ``max_chars``."""
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        if size and size + len(para) > max_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        # A single oversized paragraph (big table / ABNF block) is split hard.
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars:]
            chunks.append("\n\n".join([*buf, head]) if buf else head)
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _anchor(section: str) -> str:
    if section[0].isdigit():
        return f"section-{section}"
    if section[0].isupper():
        return f"appendix-{section}"
    return section.replace(" ", "-")


def chunk_rfc(
    text: str,
    rfc: int,
    rfc_title: str,
    max_chars: int = 1800,
    min_chars: int = 50,
) -> list[Chunk]:
    """Chunk one RFC document into section-anchored retrieval units."""
    chunks: list[Chunk] = []
    for section in parse_sections(text):
        paras = _paragraphs(section.lines)
        if not paras:
            continue
        for position, piece in enumerate(_pack(paras, max_chars)):
            if len(piece.strip()) < min_chars:
                continue
            chunks.append(
                Chunk(
                    id=f"rfc{rfc}-s{section.number.replace(' ', '_')}-{position}",
                    rfc=rfc,
                    rfc_title=rfc_title,
                    section=section.number,
                    section_title=section.title,
                    position=position,
                    text=piece,
                    url=f"https://www.rfc-editor.org/rfc/rfc{rfc}.html#{_anchor(section.number)}",
                )
            )
    return chunks
