"""Eval dataset loading and label validation.

Labels are (rfc, section) pairs. A retrieved chunk matches a label if it is
that section or any subsection of it. ``validate_labels`` guarantees every
label points at a section that actually exists in the chunked corpus, so the
dataset can never silently rot as the manifest evolves.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from specsage.models import Chunk

DATASET_PATH = Path("evals/dataset/qa.jsonl")


class Label(BaseModel):
    rfc: int
    section: str


class EvalQuestion(BaseModel):
    id: str
    scope: str  # "in" | "out"
    question: str
    labels: list[Label] = Field(default_factory=list)


def load_dataset(path: Path = DATASET_PATH) -> list[EvalQuestion]:
    questions = [
        EvalQuestion.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in dataset")
    for q in questions:
        if q.scope == "in" and not q.labels:
            raise ValueError(f"{q.id}: in-scope question without labels")
        if q.scope == "out" and q.labels:
            raise ValueError(f"{q.id}: out-of-scope question must not have labels")
    return questions


def section_matches(chunk_section: str, label_section: str) -> bool:
    return chunk_section == label_section or chunk_section.startswith(label_section + ".")


def chunk_matches_any(chunk: Chunk, labels: list[Label]) -> bool:
    return any(
        chunk.rfc == label.rfc and section_matches(chunk.section, label.section)
        for label in labels
    )


def validate_labels(questions: list[EvalQuestion], chunks: list[Chunk]) -> list[str]:
    """Return human-readable errors for labels matching no chunk in the corpus."""
    errors = []
    for q in questions:
        for label in q.labels:
            if not any(
                c.rfc == label.rfc and section_matches(c.section, label.section) for c in chunks
            ):
                errors.append(f"{q.id}: no chunk matches RFC {label.rfc} §{label.section}")
    return errors
