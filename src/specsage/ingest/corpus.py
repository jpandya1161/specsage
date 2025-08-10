"""Load the fetched corpus and produce chunks for indexing."""

import json
import logging
from pathlib import Path

from specsage.ingest.chunker import chunk_rfc
from specsage.models import Chunk

logger = logging.getLogger(__name__)


def load_corpus_chunks(rfc_dir: Path) -> list[Chunk]:
    """Chunk every RFC listed in ``<rfc_dir>/index.json``."""
    index_path = rfc_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found — run `specsage fetch` first")

    index = json.loads(index_path.read_text())
    chunks: list[Chunk] = []
    for entry in index["rfcs"]:
        path = rfc_dir / entry["path"]
        if not path.exists():
            logger.warning("missing %s, skipping", path)
            continue
        text = path.read_text(errors="replace")
        rfc_chunks = chunk_rfc(text, rfc=entry["rfc"], rfc_title=entry.get("title", ""))
        chunks.extend(rfc_chunks)
    logger.info("chunked %d RFCs into %d chunks", len(index["rfcs"]), len(chunks))
    return chunks
