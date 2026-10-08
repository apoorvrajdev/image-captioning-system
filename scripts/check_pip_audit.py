"""Gate a pip-audit JSON report against the reviewed baseline of accepted findings.

Usage:
    pip-audit -r requirements.txt -f json -o audit.json   # exits 1 whenever it finds anything
    python -m scripts.check_pip_audit audit.json --baseline .github/pip-audit-baseline.txt

pip-audit fails on every finding, including the ones TASK-020 reviewed and accepted
(ADR-025). This gate prints every finding, marked as in the baseline or new, and exits
non-zero when:

* a finding's (package, version, vulnerability) isn't in the baseline;
* a baseline entry matches no finding, so an exception can't outlive its reason;
* pip-audit skipped a dependency it couldn't audit;
* the report or the baseline is missing or malformed.

A baseline entry matches exactly one finding: the same normalised package name, the
same version string and the same vulnerability id. Aliases are printed but never matched,
so a renamed or newly split advisory fails the gate until someone reviews it. The script
uses only the standard library, so CI can run it on a bare interpreter (ADR-026).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    vuln_id: str
    aliases: frozenset[str] = field(default_factory=frozenset)
    fix_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineEntry:
    package: str
    version: str
    vuln_id: str
    line: int


@dataclass(frozen=True)
class GateResult:
    accepted: list[Finding]
    new: list[Finding]
    stale: list[BaselineEntry]
    unaudited: list[str]

    @property
    def passed(self) -> bool:
        return not (self.new or self.stale or self.unaudited)


def normalize(name: str) -> str:
    """PEP 503 name normalisation, so ``Python_Multipart`` matches ``python-multipart``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_baseline(text: str) -> list[BaselineEntry]:
    """Parse ``<package> <version> <vulnerability id>`` lines; ``#`` starts a comment."""
    entries: list[BaselineEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"baseline line {number}: expected '<package> <version> <vulnerability id>', got {raw!r}"
            )
        entry = BaselineEntry(normalize(parts[0]), parts[1], parts[2], number)
        key = (entry.package, entry.version, entry.vuln_id)
        if key in seen:
            raise ValueError(f"baseline line {number}: duplicate entry {' '.join(key)}")
        seen.add(key)
        entries.append(entry)
    return entries


def parse_report(report: Any) -> tuple[list[Finding], list[str]]:
    """Return the report's findings, de-duplicated, and the dependencies it couldn't audit."""
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("report has no audited dependencies; is it pip-audit's JSON output?")
    findings: dict[tuple[str, str, str], Finding] = {}
    unaudited: list[str] = []
    for dependency in dependencies:
        package = normalize(str(dependency["name"]))
        if "skip_reason" in dependency:
            version = str(dependency.get("version", "?"))
            unaudited.append(f"{package} {version}: {dependency['skip_reason']}")
            continue
        # pip-audit writes both keys for every dependency it audited. Defaulting either one
        # would read a changed or truncated report as clean.
        version = str(dependency["version"])
        vulns = dependency["vulns"]
        if not isinstance(vulns, list):
            raise ValueError(f"{package}: 'vulns' is {type(vulns).__name__}, not a list")
        for vuln in vulns:
            finding = Finding(
                package,
                version,
                str(vuln["id"]),
                frozenset(str(alias) for alias in vuln.get("aliases", [])),
                tuple(str(fix) for fix in vuln.get("fix_versions", [])),
            )
            findings.setdefault((package, version, finding.vuln_id), finding)
    return list(findings.values()), unaudited


def _key(finding: Finding) -> tuple[str, str, str]:
    return (finding.package, finding.version, finding.vuln_id)


def _matches(entry: BaselineEntry, finding: Finding) -> bool:
    return (entry.package, entry.version, entry.vuln_id) == _key(finding)


def evaluate(
    findings: Sequence[Finding], unaudited: Sequence[str], baseline: Sequence[BaselineEntry]
) -> GateResult:
    accepted = [f for f in findings if any(_matches(e, f) for e in baseline)]
    new = [f for f in findings if f not in accepted]
    stale = [e for e in baseline if not any(_matches(e, f) for f in findings)]
    return GateResult(accepted, new, stale, list(unaudited))


def render(result: GateResult, dependencies: int) -> list[str]:
    audited = dependencies - len(result.unaudited)
    total = len(result.accepted) + len(result.new)
    lines = [
        f"pip-audit: {audited} dependencies audited, {total} findings "
        f"({len(result.accepted)} in the baseline, {len(result.new)} new)."
    ]
    for finding in sorted(result.accepted + result.new, key=_key):
        status = "NEW" if finding in result.new else "baseline"
        fix = ", ".join(finding.fix_versions) or "none"
        aliases = ", ".join(sorted(finding.aliases)) or "none"
        lines.append(
            f"  [{status}] {finding.package} {finding.version} {finding.vuln_id} "
            f"(aliases: {aliases}; fix: {fix})"
        )
    for finding in sorted(result.new, key=_key):
        lines.append(
            f"::error::New vulnerability {finding.vuln_id} in {finding.package} {finding.version}. "
            "Upgrade the package, or review it and add it to the baseline with its reason."
        )
    for entry in result.stale:
        lines.append(
            f"::error::Baseline line {entry.line} ({entry.package} {entry.version} {entry.vuln_id}) "
            "matches no finding. Remove it and its docs/SECURITY.md entry."
        )
    for skipped in result.unaudited:
        lines.append(f"::error::Dependency not audited: {skipped}")
    lines.append("Gate passed." if result.passed else "Gate failed.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="pip-audit JSON report (-f json -o <file>)")
    parser.add_argument("--baseline", type=Path, required=True, help="accepted findings file")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        findings, unaudited = parse_report(report)
        baseline = parse_baseline(args.baseline.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"::error::pip-audit gate could not run: {exc}")
        return 1
    result = evaluate(findings, unaudited, baseline)
    print("\n".join(render(result, len(report["dependencies"]))))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
