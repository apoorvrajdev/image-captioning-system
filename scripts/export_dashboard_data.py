"""Export the Phase 3 dashboard data from committed results.

Usage:
    python -m scripts.export_dashboard_data            # rewrite the JSON
    python -m scripts.export_dashboard_data --check    # fail if it is stale

Reads ``results/phase3-comparison/comparison.json`` and every
``results/phase3-latency-*/latency.json``, checks that they agree (slice,
protocol, settings, model identity), and writes one JSON file that the SPA
imports at build time (ADR-021). Values are copied verbatim from the results.

The output is generated: don't edit it by hand. Re-run this after committing a
new comparison summary or latency run. ``--check`` writes nothing and exits
non-zero if the file differs from a fresh export.
"""

from __future__ import annotations

from pathlib import Path

import click

from captioning.evaluation.dashboard import (
    DEFAULT_COMPARISON_ID,
    DEFAULT_LATENCY_PREFIX,
    EXPORT_COMMAND,
    DashboardExportError,
    build_dashboard_data,
    render_dashboard_json,
)

DEFAULT_OUTPUT = Path("frontend/src/generated/phase3-dashboard.json")


@click.command()
@click.option(
    "--results-root",
    default=Path("results"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory holding the committed run directories.",
)
@click.option(
    "--comparison-id",
    default=DEFAULT_COMPARISON_ID,
    show_default=True,
    help="Quality summary directory under --results-root.",
)
@click.option(
    "--latency-prefix",
    default=DEFAULT_LATENCY_PREFIX,
    show_default=True,
    help="Every directory under --results-root with this prefix is a latency run.",
)
@click.option(
    "--output",
    default=DEFAULT_OUTPUT,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="The dashboard JSON the SPA imports.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Write nothing; exit non-zero if --output differs from a fresh export.",
)
def main(
    results_root: Path, comparison_id: str, latency_prefix: str, output: Path, check: bool
) -> None:
    """Build the dashboard data from committed results and write or check it."""
    try:
        data = build_dashboard_data(
            results_root, comparison_id=comparison_id, latency_prefix=latency_prefix
        )
    except DashboardExportError as exc:
        raise click.ClickException(str(exc)) from exc
    text = render_dashboard_json(data)
    runs = sum(len(m["source_run_ids"]) for m in data["models"])

    if check:
        current = output.read_text(encoding="utf-8") if output.is_file() else None
        if current != text:
            raise click.ClickException(
                f"{output.as_posix()} is out of date with {results_root.as_posix()}; "
                f"run {EXPORT_COMMAND}"
            )
        click.echo(f"{output.as_posix()} is up to date ({len(data['models'])} models, {runs} runs)")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    click.echo(f"Exported {len(data['models'])} models from {runs} runs -> {output.as_posix()}")


if __name__ == "__main__":
    main()
