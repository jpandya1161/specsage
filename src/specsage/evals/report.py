"""Render eval results as a self-contained HTML report (plain semantic tables)."""

import html
from typing import Any

_CSS = """
body { font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
       background: #f5f6f3; color: #1a1f24; max-width: 60rem;
       margin: 0 auto; padding: 2rem 1.25rem; font-size: 0.85rem; line-height: 1.5; }
h1 { font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;
     border-bottom: 1px solid #1a1f24; padding-bottom: 0.5rem; }
h2 { font-size: 0.95rem; margin-top: 2.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; background: #fdfdfb; }
th, td { border: 1px solid #d9dcd3; padding: 0.35rem 0.6rem; text-align: left; }
th { background: #eceee8; font-weight: 700; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.good { color: #1e6f3e; } .bad { color: #b3382d; }
.dim { color: #6b7168; }
details { margin: 0.5rem 0; }
summary { cursor: pointer; color: #2447a8; }
"""


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return html.escape(str(v))


def _kv_table(d: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td class='num'>{_fmt(v)}</td></tr>"
        for k, v in d.items()
        if not isinstance(v, dict)
    )
    return f"<table>{rows}</table>"


def _retrieval_table(summary: dict[str, dict[str, float]]) -> str:
    stages = list(summary)
    metric_names = list(next(iter(summary.values())))
    head = "<tr><th>stage</th>" + "".join(f"<th>{m}</th>" for m in metric_names) + "</tr>"
    rows = "".join(
        f"<tr><td>{stage}</td>"
        + "".join(f"<td class='num'>{_fmt(summary[stage][m])}</td>" for m in metric_names)
        + "</tr>"
        for stage in stages
    )
    return f"<table>{head}{rows}</table>"


def _generation_rows(per_question: list[dict[str, Any]]) -> str:
    rows = []
    for r in per_question:
        ok = (r["scope"] == "in") != r["refused"]
        mark = "<span class='good'>ok</span>" if ok else "<span class='bad'>miss</span>"
        grounded = (
            f"{r.get('segments_supported', '—')}/{r.get('segments', '—')}"
            if not r["refused"]
            else "—"
        )
        rows.append(
            f"<tr><td>{r['id']}</td><td>{html.escape(r['question'][:70])}</td>"
            f"<td>{r['scope']}</td><td>{'yes' if r['refused'] else 'no'}</td>"
            f"<td>{mark}</td><td class='num'>{r.get('citations', '—')}</td>"
            f"<td class='num'>{grounded}</td><td class='num'>{r['seconds']}s</td></tr>"
        )
    head = (
        "<tr><th>id</th><th>question</th><th>scope</th><th>refused</th>"
        "<th>verdict</th><th>citations</th><th>grounded segs</th><th>time</th></tr>"
    )
    return f"<table>{head}{''.join(rows)}</table>"


def render_report(results: dict[str, Any]) -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>specsage — evaluation report</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>specsage evaluation report</h1>",
        f"<p class='dim'>ran at {html.escape(results['ran_at'])}</p>",
        "<h2>Configuration</h2>",
        _kv_table(results["config"]),
        "<h2>Retrieval quality by stage (in-scope questions)</h2>",
        "<p class='dim'>precision@5 / recall@5 over labeled relevant sections; "
        "MRR = mean reciprocal rank of the first relevant chunk.</p>",
        _retrieval_table(results["retrieval"]["summary"]),
    ]
    if "generation" in results:
        gen = results["generation"]
        parts += [
            "<h2>Generation quality</h2>",
            _kv_table(gen["summary"]),
            "<h2>Per-question outcomes</h2>",
            _generation_rows(gen["per_question"]),
        ]
    parts.append("</body></html>")
    return "".join(parts)
