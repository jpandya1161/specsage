"""Download the pinned RFC corpus from rfc-editor.org.

For each RFC in the manifest we fetch the plain-text document and the
rfc-editor JSON metadata (title, date, status). Results are cached on disk,
so re-running only downloads what is missing. A corpus index is written to
``<out>/index.json`` and is the single source of truth for corpus size.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.rfc-editor.org/rfc"
MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, list[int]]:
    data = json.loads(path.read_text())
    return data["families"]


def _fetch_one(client: httpx.Client, number: int, out_dir: Path) -> dict[str, Any] | None:
    """Fetch one RFC's text + metadata. Returns index entry or None on failure."""
    txt_path = out_dir / f"rfc{number}.txt"
    meta: dict[str, Any] = {}

    meta_resp = client.get(f"{BASE_URL}/rfc{number}.json")
    if meta_resp.status_code == 200:
        raw = meta_resp.json()
        meta = {
            "title": raw.get("title", ""),
            "date": f"{raw.get('pub_month', '')} {raw.get('pub_year', '')}".strip(),
            "status": raw.get("status", ""),
            "obsoleted_by": raw.get("obsoleted_by", []),
        }
    else:
        logger.warning("rfc%d: metadata fetch failed (%d)", number, meta_resp.status_code)

    if not txt_path.exists():
        txt_resp = client.get(f"{BASE_URL}/rfc{number}.txt")
        if txt_resp.status_code != 200:
            logger.error("rfc%d: text fetch failed (%d)", number, txt_resp.status_code)
            return None
        txt_path.write_text(txt_resp.text)

    return {"rfc": number, **meta, "path": txt_path.name}


def fetch_corpus(out_dir: Path, delay_s: float = 0.2) -> dict[str, Any]:
    """Fetch every manifest RFC into ``out_dir``; returns the corpus index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    families = load_manifest()

    entries: list[dict[str, Any]] = []
    failures: list[int] = []
    seen: set[int] = set()

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for family, numbers in families.items():
            for number in numbers:
                if number in seen:
                    continue
                seen.add(number)
                entry = _fetch_one(client, number, out_dir)
                if entry is None:
                    failures.append(number)
                    continue
                entry["family"] = family
                entries.append(entry)
                time.sleep(delay_s)

    index = {"count": len(entries), "failures": failures, "rfcs": entries}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    logger.info("fetched %d RFCs (%d failures)", len(entries), len(failures))
    return index
