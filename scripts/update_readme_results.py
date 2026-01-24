"""Inject eval results into README.md between the EVAL RESULTS markers.

Usage: uv run python scripts/update_readme_results.py

Reads evals/results/results.json (produced by `specsage eval`) and rewrites
the README's results block. This is the only way numbers enter the README.
"""

import json
import re
from pathlib import Path

README = Path("README.md")
RESULTS = Path("evals/results/results.json")
BEGIN = "<!-- EVAL-RESULTS:BEGIN (filled from evals/results by this script — do not edit) -->"
END = "<!-- EVAL-RESULTS:END -->"


def retrieval_table(summary: dict) -> str:
    lines = [
        "| retrieval stage | precision@5 | recall@5 | MRR |",
        "|---|---|---|---|",
    ]
    for stage, m in summary.items():
        lines.append(
            f"| {stage} | {m['precision@5']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} |"
        )
    return "\n".join(lines)


def generation_table(summary: dict) -> str:
    rows = {
        "in-scope answered (not wrongly refused)": summary["in_scope_answer_rate"],
        "out-of-scope refused": summary["out_of_scope_refusal_rate"],
        "citation validity (markers resolving to real sources)": summary["citation_validity"],
        f"groundedness (supported claim segments, n={summary['segments_scored']})": summary[
            "groundedness"
        ],
    }
    lines = ["| end-to-end metric | value |", "|---|---|"]
    for name, value in rows.items():
        lines.append(f"| {name} | {value:.3f} |" if value is not None else f"| {name} | — |")
    return "\n".join(lines)


def main() -> None:
    results = json.loads(RESULTS.read_text())
    cfg = results["config"]
    parts = [
        BEGIN,
        "",
        f"*Run of {results['ran_at']} — {cfg['llm_provider']}:{cfg['llm_model']}, "
        f"{cfg['questions_in_scope']} in-scope / {cfg['questions_out_of_scope']} "
        "out-of-scope questions.*",
        "",
        retrieval_table(results["retrieval"]["summary"]),
    ]
    if "generation" in results:
        parts += ["", generation_table(results["generation"]["summary"])]
    parts += ["", END]
    block = "\n".join(parts)

    text = README.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit("README markers not found")
    README.write_text(pattern.sub(block, text))
    print("README results block updated")


if __name__ == "__main__":
    main()
