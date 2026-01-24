"""Chunker tests against a realistic legacy-format RFC fixture."""

import textwrap

from specsage.ingest.chunker import chunk_rfc, normalize, parse_sections

# A miniature RFC in the legacy paginated format: running footer/header,
# form feed, TOC with dot leaders, references section, appendix.
FIXTURE = textwrap.dedent("""\
    Internet Engineering Task Force (IETF)                        A. Example
    Request for Comments: 9999                                  October 2024
    Category: Standards Track


                        The Example Transfer Protocol (ETP)

    Abstract

       This document defines the Example Transfer Protocol, a protocol
       used only inside specsage's test suite.

    Status of This Memo

       This is an Internet Standards Track document.

    Table of Contents

       1.  Introduction  . . . . . . . . . . . . . . . . . . . . . . .   2
       2.  Message Format  . . . . . . . . . . . . . . . . . . . . . .   2
         2.1.  Header Fields . . . . . . . . . . . . . . . . . . . . .   3
       3.  Security Considerations  . . . . . . . . . . . . . . . . . .  4

    1.  Introduction

       The Example Transfer Protocol (ETP) moves example payloads between
       consenting test fixtures.  It exists so that the chunker has
       something realistic to parse.

    2.  Message Format

       An ETP message consists of a fixed header followed by a variable
       payload.

    Example, et al.              Standards Track                    [Page 1]
    \fRFC 9999                          ETP                       October 2024

    2.1.  Header Fields

       The header contains exactly three fields.  The VERSION field MUST be
       set to 1.  The LENGTH field encodes the payload size in octets.
       The CHECKSUM field is computed as described in Section 3.

    3.  Security Considerations

       ETP provides no confidentiality.  Deployments MUST run ETP over a
       transport providing encryption, such as TLS [RFC8446].

    4.  References

    4.1.  Normative References

       [RFC8446]  Rescorla, E., "TLS 1.3", RFC 8446, August 2018.

    Appendix A.  Example Messages

       The simplest valid ETP message is a header with LENGTH set to zero.

    Authors' Addresses

       A. Example
       Email: a@example.org
    """)


def test_normalize_strips_pagination():
    lines = normalize(FIXTURE)
    joined = "\n".join(lines)
    assert "[Page 1]" not in joined
    assert "October 2024" not in joined.split("Abstract")[1]  # running header removed
    assert "\f" not in joined


def test_parse_sections_finds_real_headings_only():
    sections = parse_sections(FIXTURE)
    numbers = [s.number for s in sections]
    assert "1" in numbers
    assert "2" in numbers
    assert "2.1" in numbers
    assert "3" in numbers
    assert "A" in numbers
    assert "abstract" in numbers
    # Skipped boilerplate:
    assert not any("reference" in s.title.lower() for s in sections)
    assert not any("address" in s.title.lower() for s in sections)
    assert not any("status of this memo" in s.title.lower() for s in sections)


def test_toc_entries_are_not_sections():
    sections = parse_sections(FIXTURE)
    # TOC lists "2.1. Header Fields ... 3" — the real 2.1 must appear exactly once
    assert sum(1 for s in sections if s.number == "2.1") == 1
    intro = next(s for s in sections if s.number == "1")
    assert "consenting test fixtures" in "\n".join(intro.lines)


def test_chunks_carry_provenance():
    chunks = chunk_rfc(FIXTURE, rfc=9999, rfc_title="The Example Transfer Protocol")
    assert chunks, "expected at least one chunk"
    by_section = {c.section for c in chunks}
    assert "2.1" in by_section
    header_chunk = next(c for c in chunks if c.section == "2.1")
    assert header_chunk.id.startswith("rfc9999-s2.1-")
    assert header_chunk.url.endswith("#section-2.1")
    assert "VERSION field MUST" in header_chunk.text
    assert "RFC 9999" in header_chunk.embed_text
    appendix = next(c for c in chunks if c.section == "A")
    assert appendix.url.endswith("#appendix-A")


def test_page_break_does_not_split_section_content():
    chunks = chunk_rfc(FIXTURE, rfc=9999, rfc_title="ETP")
    fmt = next(c for c in chunks if c.section == "2")
    assert "variable" in fmt.text  # paragraph spanning the page break survives


def test_long_sections_split_into_multiple_chunks():
    body = "\n\n".join(
        f"   Paragraph {i} about the ETP protocol behavior in some detail. " * 4
        for i in range(30)
    )
    doc = f"1.  Big Section\n\n{body}\n"
    chunks = chunk_rfc(doc, rfc=1, rfc_title="Big", max_chars=800)
    assert len(chunks) > 3
    assert all(len(c.text) <= 800 + 100 for c in chunks)
    positions = [c.position for c in chunks]
    assert positions == sorted(positions)
