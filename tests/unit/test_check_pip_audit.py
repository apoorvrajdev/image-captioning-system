"""Tests for the pip-audit baseline gate (TASK-021, ADR-026).

Reports are hand-built in pip-audit's JSON shape, so no audit runs and nothing is
downloaded. The last test checks the committed baseline itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import scripts.check_pip_audit as gate

REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE = """
# keras: accepted until the TensorFlow / Keras migration.
keras 2.15.0 PYSEC-2025-76
Protobuf 4.25.9 PYSEC-2026-1805   # names are normalised
"""


def _dependency(name: str, version: str, *vulns: tuple[str, list[str]]) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "vulns": [
            {"id": vid, "aliases": aliases, "fix_versions": ["9.9.9"], "description": ""}
            for vid, aliases in vulns
        ],
    }


def _report(*dependencies: dict[str, Any]) -> dict[str, Any]:
    return {"dependencies": [_dependency("fastapi", "0.133.0"), *dependencies], "fixes": []}


_BASELINED = (
    _dependency("keras", "2.15.0", ("PYSEC-2025-76", [])),
    _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
)


def _run(report: dict[str, Any], baseline: str = BASELINE) -> gate.GateResult:
    findings, unaudited = gate.parse_report(report)
    return gate.evaluate(findings, unaudited, gate.parse_baseline(baseline))


def test_baselined_findings_pass_and_duplicates_collapse() -> None:
    keras = ("PYSEC-2025-76", ["CVE-2025-9906"])
    result = _run(
        _report(
            _dependency("keras", "2.15.0", keras, keras),
            _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
        )
    )
    assert result.passed
    assert [f.vuln_id for f in result.accepted] == ["PYSEC-2025-76", "PYSEC-2026-1805"]


def test_a_finding_outside_the_baseline_fails() -> None:
    result = _run(
        _report(
            _dependency("keras", "2.15.0", ("PYSEC-2025-76", [])),
            _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
            _dependency("starlette", "1.3.1", ("PYSEC-2099-1", [])),
        )
    )
    assert not result.passed
    assert [(f.package, f.vuln_id) for f in result.new] == [("starlette", "PYSEC-2099-1")]
    assert result.stale == []


def test_an_entry_is_pinned_to_its_version() -> None:
    # The same advisory on another version is new, and the old entry goes stale.
    result = _run(
        _report(
            _dependency("keras", "2.16.0", ("PYSEC-2025-76", [])),
            _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
        )
    )
    assert [(f.package, f.version) for f in result.new] == [("keras", "2.16.0")]
    assert [(e.package, e.version) for e in result.stale] == [("keras", "2.15.0")]


def test_an_alias_does_not_stand_in_for_the_id() -> None:
    # One baseline line accepts one advisory record; a record that only lists the id as an
    # alias (a split or renamed advisory) is new until it's reviewed.
    result = _run(
        _report(
            _dependency("keras", "2.15.0", ("PYSEC-2025-76", []), ("GHSA-new", ["PYSEC-2025-76"])),
            _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
        )
    )
    assert [f.vuln_id for f in result.new] == ["GHSA-new"]
    assert result.stale == []


@pytest.mark.parametrize(
    "report",
    [
        {"dependencies": ["keras"]},
        {"dependencies": [{"name": "keras", "vulns": None}]},
        # Next to findings that pass, so only the shape check can fail these. A dependency
        # pip-audit audited always has a version and a list of vulns.
        _report(*_BASELINED, {"name": "starlette", "version": "1.3.1"}),
        _report(*_BASELINED, {"name": "starlette", "version": "1.3.1", "vulns": {}}),
        _report(*_BASELINED, {"name": "starlette", "vulns": []}),
    ],
    ids=[
        "dependency-not-object",
        "vulns-null",
        "vulns-missing",
        "vulns-not-list",
        "version-missing",
    ],
)
def test_main_reports_an_unexpected_report_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], report: Any
) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(BASELINE, encoding="utf-8")
    assert gate.main([str(path), "--baseline", str(baseline)]) == 1
    assert "could not run" in capsys.readouterr().out


def test_a_stale_entry_fails() -> None:
    result = _run(_report(_dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", []))))
    assert not result.passed
    assert [e.vuln_id for e in result.stale] == ["PYSEC-2025-76"]


def test_an_unaudited_dependency_fails() -> None:
    report = _report(
        _dependency("keras", "2.15.0", ("PYSEC-2025-76", [])),
        _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
    )
    report["dependencies"].append({"name": "private-pkg", "skip_reason": "not on PyPI"})
    result = _run(report)
    assert not result.passed
    assert result.unaudited == ["private-pkg ?: not on PyPI"]
    lines = gate.render(result, len(report["dependencies"]))
    assert lines[0].startswith("pip-audit: 3 dependencies audited")


@pytest.mark.parametrize(
    "baseline",
    ["keras 2.15.0\n", "keras PYSEC-2025-76\n", "keras 2.15.0 A\nkeras 2.15.0 A\n"],
    ids=["missing-id", "missing-version", "duplicate"],
)
def test_a_malformed_baseline_is_rejected(baseline: str) -> None:
    with pytest.raises(ValueError):
        gate.parse_baseline(baseline)


@pytest.mark.parametrize(
    "report", [{}, {"dependencies": []}, []], ids=["no-key", "empty", "not-dict"]
)
def test_a_report_without_dependencies_is_rejected(report: Any) -> None:
    with pytest.raises(ValueError):
        gate.parse_report(report)


def test_main_prints_every_finding_and_fails_on_new_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "audit.json"
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(BASELINE, encoding="utf-8")
    report.write_text(
        json.dumps(
            _report(
                _dependency("keras", "2.15.0", ("PYSEC-2025-76", [])),
                _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
            )
        ),
        encoding="utf-8",
    )
    assert gate.main([str(report), "--baseline", str(baseline)]) == 0
    passed = capsys.readouterr().out
    assert "[baseline] keras 2.15.0 PYSEC-2025-76 (aliases: none; fix: 9.9.9)" in passed
    assert passed.rstrip().endswith("Gate passed.")

    report.write_text(
        json.dumps(
            _report(
                _dependency("keras", "2.15.0", ("PYSEC-2025-76", [])),
                _dependency("protobuf", "4.25.9", ("PYSEC-2026-1805", [])),
                _dependency("pillow", "12.3.0", ("PYSEC-2099-2", [])),
            )
        ),
        encoding="utf-8",
    )
    assert gate.main([str(report), "--baseline", str(baseline)]) == 1
    failed = capsys.readouterr().out
    assert "[NEW] pillow 12.3.0 PYSEC-2099-2" in failed
    assert "::error::New vulnerability PYSEC-2099-2 in pillow 12.3.0" in failed


def test_main_fails_when_the_report_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(BASELINE, encoding="utf-8")
    assert gate.main([str(tmp_path / "absent.json"), "--baseline", str(baseline)]) == 1
    assert "could not run" in capsys.readouterr().out


def test_the_committed_baseline_parses_and_covers_only_the_reviewed_packages() -> None:
    # A new package in the baseline is a policy change: update ADR-026 and SECURITY.md first.
    entries = gate.parse_baseline(
        (REPO_ROOT / ".github" / "pip-audit-baseline.txt").read_text(encoding="utf-8")
    )
    assert {(e.package, e.version) for e in entries} == {
        ("keras", "2.15.0"),
        ("protobuf", "4.25.9"),
    }
    assert len(entries) == 14
