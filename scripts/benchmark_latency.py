"""Time one model's captioning latency on the first images of the Phase 3 slice.

Usage:
    # A Hugging Face baseline (needs the [hf] extra and the slice images):
    python -m scripts.benchmark_latency \\
        --config configs/base.yaml \\
        --images-dir /path/to/coco2017/train2017 \\
        --model blip-base --device cpu --environment "<host and hardware>"

    # The CNN + Transformer:
    python -m scripts.benchmark_latency \\
        --config configs/base.yaml \\
        --images-dir /path/to/coco2017/train2017 \\
        --model inceptionv3-transformer-stabilized \\
        --cnn-weights <checkpoint>/model.h5 --cnn-tokenizer-dir <checkpoint> \\
        --device cpu --environment "<host and hardware>"

Each invocation times one model and writes a new
``results/<prefix><model_id>-<decoding>-<device>/latency.json``
(``docs/EVAL_METHODOLOGY.md`` § 9). The model is selected, built and checked
exactly as in ``scripts/compare_models.py``, through the ``captioning.baselines``
adapters, and timed by ``captioning.evaluation.latency``. No metrics,
predictions or other quality-evaluation files are written.

Before any model loads, the script checks the benchmark settings, the slice's
image and reference counts, the model id, that the run directory doesn't exist
yet, and that every benchmark image is present.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from captioning.config import load_config
from captioning.evaluation import (
    LatencySettings,
    load_eval_slice,
    measure_latency,
    runtime_info,
    slice_fingerprint,
    summarize,
    time_load,
)
from captioning.evaluation.latency import CLOCK_NAME, LATENCY_PROTOCOL, Clock
from captioning.utils import configure_logging, get_logger, set_global_seed
from scripts.compare_models import PROTOCOL_SLICE, _check_identity, build_captioner, resolve_models

log = get_logger(__name__)

LATENCY_FILE = "latency.json"
# Read at call time, so tests can substitute a scripted clock.
CLOCK: Clock = time.perf_counter
TIMING = {
    "clock": CLOCK_NAME,
    "sample": (
        "one Captioner.caption() call on one batch: image reading and decoding, preprocessing, "
        "generation, token decoding and caption normalisation"
    ),
    "load": "captioner construction plus load(), timed once and never part of a sample",
}


def tensorflow_gpus() -> list[str]:
    """Names of the GPUs TensorFlow can see. TensorFlow is already loaded by the CNN."""
    import tensorflow as tf

    return [device.name for device in tf.config.list_physical_devices("GPU")]


def _check_cnn_device(device: str, gpus: list[str]) -> None:
    # TensorFlow places the CNN + Transformer itself, so --device can only be
    # checked against what it can see, not passed to it.
    if device == "cpu" and gpus:
        raise click.ClickException(
            f"--device cpu, but TensorFlow can see {', '.join(gpus)} and would run the CNN there; "
            'hide them with CUDA_VISIBLE_DEVICES="" or use --device cuda'
        )
    if device == "cuda" and not gpus:
        raise click.ClickException(
            "--device cuda, but TensorFlow can see no GPU and would run the CNN on the CPU"
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
@click.option("--model", "model_id", required=True, help="Model id from config.compare to time.")
@click.option(
    "--device",
    required=True,
    type=click.Choice(["cpu", "cuda"]),
    help="Device the model runs on. Hugging Face models are moved there; for the CNN it is "
    "checked against the GPUs TensorFlow can see.",
)
@click.option(
    "--environment",
    required=True,
    help='Name of the host and hardware, recorded verbatim (e.g. "Kaggle, 1x Tesla T4").',
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
    "--num-images",
    default=32,
    show_default=True,
    type=int,
    help="Benchmark inputs: this many images from the start of the slice, in slice order.",
)
@click.option(
    "--batch-size",
    "batch_sizes",
    default=(1, 8),
    show_default=True,
    multiple=True,
    type=int,
    help="Batch size to time. Repeat for several; each must divide --num-images.",
)
@click.option("--warmup-passes", default=1, show_default=True, type=int)
@click.option("--measured-passes", default=5, show_default=True, type=int)
@click.option(
    "--results-root", default=Path("results"), show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--run-prefix",
    default="phase3-latency-",
    show_default=True,
    help="Run directory name is <prefix><model_id>-<decoding>-<device>.",
)
@click.option(
    "--cnn-weights", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--cnn-tokenizer-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def main(  # — CLI option count is unavoidable
    config_path: Path,
    images_dir: Path,
    model_id: str,
    device: str,
    environment: str,
    slice_path: Path,
    expected_images: int,
    expected_references: int,
    num_images: int,
    batch_sizes: tuple[int, ...],
    warmup_passes: int,
    measured_passes: int,
    results_root: Path,
    run_prefix: str,
    cnn_weights: Path | None,
    cnn_tokenizer_dir: Path | None,
) -> None:
    """Time one model on the first slice images and write its latency.json."""
    configure_logging()
    if not environment.strip():
        raise click.ClickException("--environment must name the host and hardware")
    try:
        settings = LatencySettings(
            num_images=num_images,
            batch_sizes=tuple(sorted(batch_sizes)),
            warmup_passes=warmup_passes,
            measured_passes=measured_passes,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

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
    if settings.num_images > n_images:
        raise click.ClickException(
            f"--num-images {settings.num_images} exceeds the slice's {n_images} images"
        )

    (spec,) = resolve_models(config, [model_id], results_root, run_prefix)
    if spec.backend == "cnn" and (cnn_weights is None or cnn_tokenizer_dir is None):
        raise click.ClickException("the CNN needs --cnn-weights and --cnn-tokenizer-dir")
    run_dir = spec.run_dir.with_name(f"{spec.run_dir.name}-{device}")
    if run_dir.exists():
        raise click.ClickException(f"run directory already exists: {run_dir}")
    image_paths = list(eval_slice.image_paths[: settings.num_images])
    missing = [p.name for p in image_paths if not p.is_file()]
    if missing:
        raise click.ClickException(
            f"{len(missing)} of {settings.num_images} benchmark images are missing under "
            f"{images_dir}: {', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
        )

    clock = CLOCK
    captioner, load_seconds = time_load(
        lambda: build_captioner(
            spec,
            config,
            device=device,
            cnn_weights=cnn_weights,
            cnn_tokenizer_dir=cnn_tokenizer_dir,
        ),
        clock=clock,
    )
    identity = captioner.identity
    _check_identity(spec, identity)
    tf_gpus: list[str] | None = None
    if spec.backend == "cnn":
        tf_gpus = tensorflow_gpus()
        _check_cnn_device(device, tf_gpus)

    batches = measure_latency(captioner, image_paths, settings, clock=clock)

    record = {
        "protocol": LATENCY_PROTOCOL,
        "model_id": identity.model_id,
        "backend": spec.backend,
        "captioner": type(captioner).__name__,
        "hub_repo": identity.hub_repo,
        "revision": identity.revision,
        "decode_strategy": spec.decode_strategy,
        "decode_settings": dict(identity.decode_settings),
        "device": device,
        "environment": environment,
        "inputs": {
            "slice": {
                "source": slice_path.as_posix(),
                "fingerprint_sha256": slice_fingerprint(eval_slice),
                "images": n_images,
                "references": n_references,
            },
            "images": [p.name for p in image_paths],
        },
        "settings": settings.to_dict(),
        "timing": TIMING,
        "load_seconds": load_seconds,
        "batches": [batch.to_dict() for batch in batches],
        "seed": config.train.seed,
        "runtime": {**runtime_info(), "tensorflow_gpus": tf_gpus},
    }

    if run_dir.exists():
        raise click.ClickException(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / LATENCY_FILE).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    log.info("latency_run_done", model_id=identity.model_id, run_dir=str(run_dir))
    medians = ", ".join(
        f"batch {b.batch_size}: median {1000 * summarize(b.samples)['median']:.1f} ms"
        for b in batches
    )
    click.echo(f"{identity.model_id} on {device}: {run_dir} (load {load_seconds:.1f} s; {medians})")


if __name__ == "__main__":
    main()
