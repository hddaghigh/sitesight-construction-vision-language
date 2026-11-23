"""Every headline number in README.md must match the committed evidence it cites.

The README table is the project's public claim; this test reads the JSON/Markdown reports and fails
if a number there was edited by hand, a report was regenerated without updating the README, or a
row was removed. Raw per-round active-learning rows were never committed, so that row is checked
against the number printed in ``reports/m6/README.md`` instead (a weaker link, stated in the README).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def row(label: str) -> str:
    lines = [ln for ln in README.splitlines() if ln.startswith(f"| {label}")]
    assert len(lines) == 1, f"README must have exactly one table row starting with {label!r}"
    return lines[0]


def entry(entries: list[dict[str, Any]], model: str) -> dict[str, Any]:
    found = [e for e in entries if e["model"] == model]
    assert len(found) == 1, model
    return found[0]


def map50(entries: list[dict[str, Any]], model: str, protocol: str = "given_phrases") -> str:
    return f"{100 * entry(entries, model)['protocols'][protocol]['map50']:.1f}"


def test_parser_coverage_and_split_leakage() -> None:
    c = load("reports/m2/coverage_report.json")
    line = row("Caption-field parser coverage")
    assert f"{100 * c['coverage']:.2f}%" in line
    assert f"{c['n_parsed']:,} of {c['n_annotations'] - c['n_empty']:,}" in line
    k = load("reports/m2/curation_report.json")
    line = row("Leakage a random split")
    assert (
        f"{100 * k['naive_split_leakage']['test_fraction_with_train_neighbour_mean']:.1f}%" in line
    )
    assert k["grouped_split_leakage"]["test_images_with_train_neighbour"] == 0 and "0%" in line


def test_zero_shot_grounding_matches_both_reports() -> None:
    line = row("Zero-shot grounding")
    for path in ("reports/m5/val_compare.json", "reports/m4/zero_shot_val.json"):
        d = load(path)
        for model, label in (
            ("owlv2-base-patch16-ensemble", "OWLv2-base"),
            ("florence-2-large-grounding", "Florence-2-large"),
            ("florence-2-base-grounding", "Florence-2-base"),
        ):
            assert f"{label} {map50(d, model)}" in line, (path, model)


def test_fine_tuned_results_on_test() -> None:
    d = load("reports/m5/test_final.json")
    line = row("Fine-tuned grounding")
    lora = map50(d, "florence-2-base-lora-epoch_4-grounding")
    assert f"LoRA {lora} vs OWLv2 {map50(d, 'owlv2-base-patch16-ensemble')}" in line
    assert f"zero-shot Florence-2-base {map50(d, 'florence-2-base-grounding')}" in line
    lora_cap = entry(d, "florence-2-base-lora-epoch_4-regions")["fields"]
    base_cap = entry(d, "florence-2-base-regions")["fields"]
    line = row("Fine-tuned region captions")
    for key, word in (("resource", "resource"), ("colour", "colour"), ("count", "count")):
        assert f"{word} {100 * lora_cap[f'{key}_acc_loose']:.1f}%" in line
    assert all(
        f"{100 * base_cap[f'{k}_acc_loose']:.1f}%" in line for k in ("resource", "colour", "count")
    )
    ece = {
        m: entry(d, m)["protocols"]["given_phrases"]["calibration"]["ece"]
        for m in ("florence-2-base-lora-epoch_4-grounding", "owlv2-base-patch16-ensemble")
    }
    line = row("Calibration")
    assert f"{ece['florence-2-base-lora-epoch_4-grounding']:.3f}" in line
    assert f"{ece['owlv2-base-patch16-ensemble']:.3f}" in line
    assert ece["florence-2-base-lora-epoch_4-grounding"] > ece["owlv2-base-patch16-ensemble"]


def test_active_learning_numbers_match_the_m6_report() -> None:
    text = (ROOT / "reports/m6/README.md").read_text(encoding="utf-8")
    m = re.search(
        r"random ([\d.]+) ± [\d.]+;\s+uncertainty ([\d.]+) ± [\d.]+;\s+uncertainty \+ diversity ([\d.]+)",
        text,
    )
    assert m is not None
    r, u, d = m.groups()  # kept as printed: one decimal, no rounding step to disagree about
    assert f"random {r}, uncertainty {u}, uncertainty + diversity {d}" in row("Active learning")


def test_serving_benchmark() -> None:
    b = load("reports/m7/bench.json")["engines"]
    line = row("CPU serving")
    assert f"p50 {b['pytorch_fp32']['latency_cold']['p50_s']:.1f} s PyTorch" in line
    assert f"{b['onnx_fp32']['latency_cold']['p50_s']:.1f} s ONNX fp32" in line
    assert f"{b['onnx_int8']['latency_cold']['p50_s']:.1f} s int8" in line
    assert f"{b['onnx_fp32']['size_mb']:,.0f} MB vs {b['onnx_int8']['size_mb']:,.0f} MB" in line
    assert (
        f"{100 * b['onnx_int8']['agreement_with_pytorch']['identical_caption_rate']:.0f}%" in line
    )


def test_monitoring_claims_hold_in_the_demo_output() -> None:
    d = load("reports/m8/demo.json")
    line = row("Monitoring")
    assert d["same_site_false_alarm_rate"]["40"] == 0.0 and "0% false alarms" in line
    for name in ("blur_r2", "blur_strong_r5", "darker_x0.5", "rescaled_x0.5"):
        assert d["perturbations"][name]["detection_rate_full_batch"]["40"] == 1.0
        assert d["perturbations"][name]["detection_rate_25pct_degraded"]["80"] == 1.0
        assert d["perturbations"][name]["detection_rate_25pct_degraded"]["40"] < 1.0
    assert "about 80 images" in line


def test_odometry_numbers_and_evo_agreement() -> None:
    m = load("reports/m9/metrics.json")["modes"]
    line = row("RGB-D odometry")
    assert f"{100 * m['frame2frame']['ate_rmse_m']:.1f} cm (frame-to-frame)" in line
    assert f"{100 * m['keyframe']['ate_rmse_m']:.1f} cm (keyframe)" in line
    assert f"{100 * m['keyframe']['stationary_baseline_ate_rmse_m']:.0f} cm" in line
    for mode in m.values():
        assert mode["evo_ate_rmse_m"] == pytest.approx(mode["ate_rmse_m"], abs=1e-5)


def test_the_label_audit_is_still_marked_unmeasured() -> None:
    assert "not measured" in row("Label accuracy")
    assert len(load("reports/m3/audit_sample.json")["images"]) == 100


@pytest.mark.parametrize("doc", ["README.md", "TECHNICAL_REPORT.md"])
def test_relative_links_in_the_docs_point_at_files_that_exist(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
    local = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]
    assert local, "expected at least one relative link"
    missing = [t for t in local if not (ROOT / t).exists()]
    assert missing == []
