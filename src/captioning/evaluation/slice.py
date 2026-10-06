"""Load the Phase 3 evaluation slice from a committed run.

Phase 3 scores every model on the exact images and references of an existing
run instead of re-sampling COCO (``docs/EVAL_METHODOLOGY.md`` § 8.2). A run's
``predictions.jsonl`` already records that slice: one row per image, in
evaluation order, with the references its metrics were computed against.

The stored image paths belong to the machine that produced the run (Kaggle for
the committed runs), so each one is remapped by file name onto a local images
directory. Nothing else is transformed: row order, reference order and
reference text are kept exactly. Image files don't need to exist here; the
runner checks them before any model loads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalSlice:
    """The images and stored references of one committed evaluation run.

    Two slices compare equal when their image paths and references match,
    whichever ``predictions.jsonl`` they were read from.
    """

    source: Path = field(compare=False)
    image_paths: tuple[Path, ...]
    references: tuple[tuple[str, ...], ...]

    def __len__(self) -> int:
        return len(self.image_paths)


def load_eval_slice(predictions_path: str | Path, images_dir: str | Path) -> EvalSlice:
    """Read the slice recorded in a run's ``predictions.jsonl``.

    Args:
        predictions_path: A committed ``results/<run_id>/predictions.jsonl``.
        images_dir: Local directory holding the slice images. Each stored image
            path is replaced by ``images_dir / <file name>``.

    Returns:
        The slice, in file order, with references exactly as stored.

    Raises:
        ValueError: If a row lacks a non-empty ``image`` string or a non-empty
            list of reference strings. The message names the file and line.
    """
    source = Path(predictions_path)
    root = Path(images_dir)
    image_paths: list[Path] = []
    references: list[tuple[str, ...]] = []

    with source.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = row.get("image") if isinstance(row, dict) else None
            refs = row.get("references") if isinstance(row, dict) else None
            name = _file_name(image) if isinstance(image, str) else ""
            if not name:
                raise ValueError(f"{source}:{lineno}: row has no image file name")
            if not (isinstance(refs, list) and refs and all(isinstance(r, str) for r in refs)):
                raise ValueError(f"{source}:{lineno}: row needs a non-empty list of references")
            image_paths.append(root / name)
            references.append(tuple(refs))

    return EvalSlice(source=source, image_paths=tuple(image_paths), references=tuple(references))


def slice_fingerprint(eval_slice: EvalSlice) -> str:
    """SHA-256 of the slice's image file names and references, in order.

    Hashes the parsed content rather than the file bytes, so line endings and
    the machine-specific image directory don't change the identity.
    """
    payload = [
        [path.name, list(refs)]
        for path, refs in zip(eval_slice.image_paths, eval_slice.references, strict=True)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_name(stored_path: str) -> str:
    """Return the file name of a stored path written with either separator."""
    return stored_path.replace("\\", "/").rsplit("/", 1)[-1]
