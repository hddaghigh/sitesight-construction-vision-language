"""Self-contained HTML report for one monitoring check (no scripts, no external assets).

It shows the verdict first, then every test with its statistic, p-value and threshold, then how
the batch's medians compare with the reference, and how many images were individually flagged and
why. Every number in it comes from the check; nothing is estimated here. All text that came from
outside (feature names, reasons) is HTML-escaped.
"""

from __future__ import annotations

from collections import Counter
from html import escape

import numpy as np

from sitesight.monitor.drift import DriftResult

_VERDICT = {
    "ok": ("No drift detected", "ok"),
    "alert": ("DRIFT ALERT", "alert"),
    "too_small": ("Not enough data to test", "warn"),
}

_CSS = (
    "body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:60rem;padding:0 1rem}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0}"
    "th,td{border-bottom:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
    "td.n{text-align:right;font-variant-numeric:tabular-nums}"
    ".banner{padding:.8rem 1rem;border-radius:6px;font-weight:600}"
    ".ok{background:#e3f4e6;color:#14532d}.alert{background:#fde2e2;color:#7f1d1d}"
    ".warn{background:#fff3cd;color:#664d03}tr.hit td{background:#fde2e2}"
    "small{color:#555}"
)


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def render_report(
    result: DriftResult,
    reference: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    image_flags: list[list[str]],
    title: str = "SiteSight monitoring report",
    source_note: str = "",
) -> str:
    label, css = _VERDICT[result.status]
    rows = []
    for t in result.tests:
        name = f"{t.name}" + (f": {t.feature}" if t.feature else " (all features jointly)")
        rows.append(
            f'<tr class="{"hit" if t.alerted else ""}"><td>{escape(name)}</td>'
            f'<td class="n">{_fmt(t.statistic)}</td><td class="n">{_fmt(t.p_value)}</td>'
            f'<td class="n">{_fmt(t.threshold)}</td><td>{"ALERT" if t.alerted else "ok"}</td></tr>'
        )
    cmp_rows = []
    for f in reference:
        if f in batch and len(batch[f]):
            cmp_rows.append(
                f"<tr><td>{escape(f)}</td>"
                f'<td class="n">{_fmt(float(np.median(reference[f])))}</td>'
                f'<td class="n">{_fmt(float(np.median(batch[f])))}</td></tr>'
            )
    reasons = Counter(r for flags in image_flags for r in flags)
    n_flagged = sum(bool(f) for f in image_flags)
    flag_rows = "".join(
        f'<tr><td>{escape(r)}</td><td class="n">{c}</td></tr>' for r, c in reasons.most_common()
    )
    return (
        f"<!doctype html><html lang=en><meta charset=utf-8><title>{escape(title)}</title>"
        f"<style>{_CSS}</style><h1>{escape(title)}</h1>"
        f'<p class="banner {css}">{escape(label)}</p>'
        f"<p>Reference: {result.n_reference} samples. Batch: {result.n_batch} samples. "
        f"Overall false-alarm budget alpha = {_fmt(result.alpha)}. "
        f"{escape(source_note)}</p>"
        "<h2>Tests</h2><table><tr><th>test</th><th>statistic</th><th>p-value</th>"
        "<th>alert below</th><th></th></tr>" + "".join(rows) + "</table>"
        "<small>KS: largest gap between the two empirical CDFs. MMD: kernel two-sample statistic "
        "with a permutation p-value. Thresholds already include the multiple-testing split.</small>"
        "<h2>Medians</h2><table><tr><th>feature</th><th>reference</th><th>batch</th></tr>"
        + "".join(cmp_rows)
        + "</table>"
        f"<h2>Individually unusual images</h2><p>{n_flagged} of {len(image_flags)} flagged. "
        "A 1st-99th percentile band flags about 1% of normal images per feature by design.</p>"
        "<table><tr><th>reason</th><th>images</th></tr>" + flag_rows + "</table></html>"
    )
