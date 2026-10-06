"""Caption the committed Phase 3 slice with each selected model.

Usage:
    # Hugging Face baselines (needs the [hf] extra and the slice images):
    python -m scripts.compare_models \\
        --config configs/base.yaml \\
        --images-dir /path/to/coco2017/train2017 \\
        --model blip-base --model vit-gpt2 --model git-base-coco

    # The CNN + Transformer through the same harness:
    python -m scripts.compare_models \\
        --config configs/base.yaml \\
        --images-dir /path/to/coco2017/train2017 \\
        --model inceptionv3-transformer-stabilized \\
        --cnn-weights <checkpoint>/model.h5 --cnn-tokenizer-dir <checkpoint>

Each model gets a new ``results/<prefix><model_id>-<decoding>/`` holding the
five standard files from ``write_run_artifacts``, plus ``comparison_meta.json``
with the slice fingerprint, normalisation, backend and decode settings
(``docs/EVAL_METHODOLOGY.md`` § 8). Models are listed in ``config.compare``;
captioning and normalisation happen inside the ``captioning.baselines``
adapters, and metrics come from the unchanged ``compute_all_metrics``.

Before any model loads, the runner checks the slice's image and reference
counts, the selected model ids, that no output directory exists yet, and that
every slice image is present.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import click

from captioning.baselines import Captioner, CaptionerIdentity, CNNCaptioner, HFCaptioner
from captioning.config import AppConfig, BaselineDecodeConfig, ComparedModelConfig, load_config
from captioning.evaluation import (
    RunMeta,
    compute_all_metrics,
    diagnose_many,
    load_eval_slice,
    slice_fingerprint,
    write_run_artifacts,
)
from captioning.utils import configure_logging, get_logger, set_global_seed

log = get_logger(__name__)

PROTOCOL = "docs/EVAL_METHODOLOGY.md § 8"
NORMALISATION = "preprocess_caption -> strip_sentinels"
COMPARISON_META = "comparison_meta.json"
PROTOCOL_SLICE = Path("results/stabilized-greedy/predictions.jsonl")

Backend = Literal["cnn", "huggingface"]


@dataclass(frozen=True)
class ModelSpec:
    """One selected model: its config entry, backend, decoding and output directory."""

    model: ComparedModelConfig
    backend: Backend
    decode_strategy: str
    run_dir: Path


def resolve_models(
    config: AppConfig, model_ids: Sequence[str], results_root: Path, run_prefix: str
) -> list[ModelSpec]:
    """Map selected model ids to their ``config.compare`` entries and run directories."""
    compare = config.compare
    configured: dict[str, tuple[ComparedModelConfig, Backend]] = {
        compare.cnn.model_id: (compare.cnn, "cnn")
    }
    configured.update({m.model_id: (m, "huggingface") for m in compare.baselines})

    specs: list[ModelSpec] = []
    for model_id in model_ids:
        if model_id not in configured:
            raise click.ClickException(
                f"unknown model id {model_id!r}; configured: {', '.join(configured)}"
            )
        if any(spec.model.model_id == model_id for spec in specs):
            raise click.ClickException(f"model {model_id!r} is selected more than once")
        model, backend = configured[model_id]
        strategy = (
            config.serve.decode_strategy
            if backend == "cnn"
            else _hf_decode_strategy(compare.baseline_decode)
        )
        run_dir = results_root / f"{run_prefix}{model_id}-{strategy}"
        specs.append(ModelSpec(model, backend, strategy, run_dir))
    return specs


def build_captioner(
    spec: ModelSpec,
    config: AppConfig,
    *,
    device: str,
    cnn_weights: Path | None,
    cnn_tokenizer_dir: Path | None,
) -> Captioner:
    """Construct the adapter for ``spec``. Weights load later, in ``load()``."""
    if spec.backend == "huggingface":
        return HFCaptioner(spec.model, config.compare.baseline_decode, device=device)
    if cnn_weights is None or cnn_tokenizer_dir is None:
        raise click.ClickException("the CNN needs --cnn-weights and --cnn-tokenizer-dir")
    return CNNCaptioner.from_artifacts(cnn_weights, cnn_tokenizer_dir, config)


def _hf_decode_strategy(decode: BaselineDecodeConfig) -> str:
    if decode.do_sample:
        raise click.ClickException(
            "sampling isn't part of the comparison protocol (§ 8.4); set do_sample: false"
        )
    return "greedy" if decode.num_beams == 1 else "beam"


def _check_identity(spec: ModelSpec, identity: CaptionerIdentity) -> None:
    expected = (spec.model.model_id, spec.model.hub_repo, spec.model.revision)
    actual = (identity.model_id, identity.hub_repo, identity.revision)
    if actual != expected:
        raise click.ClickException(
            f"captioner identity {actual} doesn't match the config {expected}"
        )
    if (
        spec.backend == "cnn"
        and identity.decode_settings["decode_strategy"] != spec.decode_strategy
    ):
        raise click.ClickException(
            f"CNN decodes with {identity.decode_settings['decode_strategy']!r}, "
            f"but its run directory says {spec.decode_strategy!r}"
        )


def _caption_slice(captioner: Captioner, image_paths: Sequence[Path], batch_size: int) -> list[str]:
    captions: list[str] = []
    for start in range(0, len(image_paths), batch_size):
        captions.extend(captioner.caption(image_paths[start : start + batch_size]))
    return captions


def _run_meta(
    spec: ModelSpec,
    identity: CaptionerIdentity,
    n_samples: int,
    cnn_weights: Path | None,
    cnn_tokenizer_dir: Path | None,
) -> RunMeta:
    settings = identity.decode_settings
    if spec.backend == "cnn":
        return RunMeta(
            model_id=identity.model_id,
            decode_strategy=spec.decode_strategy,
            weights_path=str(cnn_weights),
            tokenizer_dir=str(cnn_tokenizer_dir),
            n_samples=n_samples,
            max_length=cast(int, settings["max_length"]),
            beam_width=cast("int | None", settings["beam_width"]),
            length_penalty=cast("float | None", settings["length_penalty"]),
            repetition_penalty=cast(float, settings["repetition_penalty"]),
        )
    pinned = f"{identity.hub_repo}@{identity.revision}"
    return RunMeta(
        model_id=identity.model_id,
        decode_strategy=spec.decode_strategy,
        weights_path=pinned,
        tokenizer_dir=pinned,
        n_samples=n_samples,
        max_length=cast(int, settings["max_new_tokens"]),
        beam_width=cast(int, settings["num_beams"]) if spec.decode_strategy == "beam" else None,
        length_penalty=None,
        repetition_penalty=cast(float, settings["repetition_penalty"]),
    )


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--images-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Local directory holding the slice images (matched by file name).",
)
@click.option(
    "--model",
    "model_ids",
    required=True,
    multiple=True,
    help="Model id from config.compare to run. Repeat for several models.",
)
@click.option(
    "--slice",
    "slice_path",
    default=PROTOCOL_SLICE,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Committed predictions.jsonl that defines the evaluation slice.",
)
@click.option("--expected-images", default=500, show_default=True, type=click.IntRange(min=1))
@click.option("--expected-references", default=732, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--results-root", default=Path("results"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--run-prefix",
    default="phase3-",
    show_default=True,
    help="Run directory name is <prefix><model_id>-<decoding>.",
)
@click.option(
    "--cnn-weights", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--cnn-tokenizer-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--batch-size", default=1, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--device", default="cpu", show_default=True, help="torch device for Hugging Face baselines."
)
@click.option(
    "--skip-meteor", is_flag=True, default=False, help="Skip METEOR (avoids needing Java)."
)
@click.option("--skip-cider", is_flag=True, default=False, help="Skip CIDEr.")
def main(  # — CLI option count is unavoidable
    config_path: Path,
    images_dir: Path,
    model_ids: tuple[str, ...],
    slice_path: Path,
    expected_images: int,
    expected_references: int,
    results_root: Path,
    run_prefix: str,
    cnn_weights: Path | None,
    cnn_tokenizer_dir: Path | None,
    batch_size: int,
    device: str,
    skip_meteor: bool,
    skip_cider: bool,
) -> None:
    """Caption the evaluation slice with each model and write one run directory per model."""
    configure_logging()
    config = load_config(config_path)
    set_global_seed(config.train.seed)

    eval_slice = load_eval_slice(slice_path, images_dir)
    n_images = len(eval_slice)
    n_references = sum(len(refs) for refs in eval_slice.references)
    if (n_images, n_references) != (expected_images, expected_references):
        raise click.ClickException(
            f"slice {slice_path} has {n_images} images and {n_references} references; "
            f"expected {expected_images} images and {expected_references} references"
        )

    specs = resolve_models(config, model_ids, results_root, run_prefix)
    if any(s.backend == "cnn" for s in specs) and (
        cnn_weights is None or cnn_tokenizer_dir is None
    ):
        raise click.ClickException("the CNN needs --cnn-weights and --cnn-tokenizer-dir")
    existing = [str(s.run_dir) for s in specs if s.run_dir.exists()]
    if existing:
        raise click.ClickException(f"run directory already exists: {', '.join(existing)}")
    missing = [p.name for p in eval_slice.image_paths if not p.is_file()]
    if missing:
        raise click.ClickException(
            f"{len(missing)} of {n_images} slice images are missing under {images_dir}: "
            f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
        )

    fingerprint = slice_fingerprint(eval_slice)
    images = [str(p) for p in eval_slice.image_paths]
    references = [list(refs) for refs in eval_slice.references]

    for spec in specs:
        captioner = build_captioner(
            spec,
            config,
            device=device,
            cnn_weights=cnn_weights,
            cnn_tokenizer_dir=cnn_tokenizer_dir,
        )
        identity = captioner.identity
        _check_identity(spec, identity)
        captioner.load()
        predictions = _caption_slice(captioner, eval_slice.image_paths, batch_size)

        metrics = compute_all_metrics(
            predictions,
            references,
            include_meteor=not skip_meteor,
            include_cider=not skip_cider,
        )
        diagnostics = diagnose_many(images, predictions, references)
        meta = _run_meta(spec, identity, n_images, cnn_weights, cnn_tokenizer_dir)

        if spec.run_dir.exists():
            raise click.ClickException(f"run directory already exists: {spec.run_dir}")
        write_run_artifacts(
            spec.run_dir,
            metrics=metrics,
            meta=meta,
            images=images,
            predictions=predictions,
            references=references,
            diagnostics=diagnostics,
        )
        comparison_meta = {
            "protocol": PROTOCOL,
            "model_id": identity.model_id,
            "backend": spec.backend,
            "captioner": type(captioner).__name__,
            "hub_repo": identity.hub_repo,
            "revision": identity.revision,
            "decode_settings": dict(identity.decode_settings),
            "normalisation": NORMALISATION,
            "slice": {
                "source": slice_path.as_posix(),
                "fingerprint_sha256": fingerprint,
                "images": n_images,
                "references": n_references,
            },
            "batch_size": batch_size,
            "device": device if spec.backend == "huggingface" else None,
            "seed": config.train.seed,
            "metrics": {"meteor": not skip_meteor, "cider": not skip_cider},
        }
        (spec.run_dir / COMPARISON_META).write_text(
            json.dumps(comparison_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        log.info("comparison_run_done", model_id=identity.model_id, run_dir=str(spec.run_dir))
        click.echo(
            f"{identity.model_id}: {spec.run_dir} "
            f"(BLEU-4 {_fmt(metrics.bleu4)}, CIDEr {_fmt(metrics.cider)})"
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
