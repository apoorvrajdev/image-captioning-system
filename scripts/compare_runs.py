"""Summarise Phase 3 run directories into one cross-model comparison.

Usage:
    python -m scripts.compare_runs \\
        results/phase3-blip-base-greedy results/phase3-vit-gpt2-greedy \\
        results/phase3-git-base-coco-greedy \\
        --reference-run results/stabilized-greedy \\
        --reference-run results/stabilized-beam-w4-lp07-rp12 \\
        --output-dir results/<comparison-id>

Positional directories are comparison-runner outputs and must contain
``comparison_meta.json``. ``--reference-run`` adds a committed run made before
the runner existed (the CNN + Transformer rows of
``docs/EVAL_METHODOLOGY.md`` § 8.4) as a labelled reference row.

Writes ``comparison.json`` (exact values) and ``comparison.md`` into a new
``--output-dir``. If any run is incomplete, malformed or scored on a different
slice, it exits non-zero, names the run, and writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from captioning.evaluation import ComparisonError, build_summary, load_run, render_markdown


@click.command()
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--reference-run",
    "reference_runs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Committed run without comparison_meta.json, shown as a reference row. Repeatable.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="New directory for comparison.json and comparison.md; must not exist.",
)
def main(run_dirs: tuple[Path, ...], reference_runs: tuple[Path, ...], output_dir: Path) -> None:
    """Check that the runs share one slice and write the comparison summary."""
    if output_dir.exists():
        raise click.ClickException(f"output directory already exists: {output_dir}")
    try:
        runs = [load_run(d) for d in run_dirs]
        runs += [load_run(d, reference=True) for d in reference_runs]
        summary = build_summary(runs)
    except ComparisonError as exc:
        raise click.ClickException(str(exc)) from exc

    output_dir.mkdir(parents=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(render_markdown(summary), encoding="utf-8")
    click.echo(f"Compared {len(summary['rows'])} runs -> {output_dir}")


if __name__ == "__main__":
    main()
