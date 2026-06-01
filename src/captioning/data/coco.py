"""Load COCO 2017 caption annotations into a ``pandas.DataFrame``.

Mirrors notebook cell 2 with two small but important upgrades that don't
change behaviour at fixed seeds:

1. **Seeded sampling.** The notebook calls ``captions.sample(120000)`` with no
   ``random_state``, so two runs produce different subsets. We thread the
   seed through so every run is identical when the seed is fixed.
2. **Path validation.** The notebook constructs paths via ``f-string`` with no
   ``os.path.exists`` check; if the dataset is missing, training fails ten
   minutes in. We check the annotations file up front and raise a clear error.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from captioning.utils.logging import get_logger

log = get_logger(__name__)

IMAGE_FILENAME_TEMPLATE = "%012d.jpg"  # Notebook cell 2: '%012d.jpg' % image_id


def load_coco_annotations(
    base_path: str | Path,
    annotations_filename: str = "captions_train2017.json",
    images_subdir: str = "train2017",
    sample_size: int = 120_000,
    seed: int | None = None,
    caption_preprocessor: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Read COCO annotations and return a (image, caption) DataFrame.

    Mirrors notebook cell 2 + cell 4 (when ``caption_preprocessor`` is
    supplied). Returns the same columns and dtypes the notebook produces.

    Args:
        base_path: Path to the COCO root containing ``annotations/`` and the
            images sub-directory.
        annotations_filename: JSON file under ``base_path / 'annotations'``.
        images_subdir: Folder containing the JPEG files.
        sample_size: Number of caption rows to keep after sampling. Use
            ``-1`` to disable sampling and keep everything.
        seed: Random seed for deterministic sampling. ``None`` matches the
            notebook's non-deterministic behaviour.
        caption_preprocessor: Optional function applied to the ``caption``
            column. Pass ``preprocessing.preprocess_caption`` to reproduce
            cell 4. Left optional so callers can stage the preprocessing
            differently (the parity audit script applies it manually).

    Returns:
        ``pd.DataFrame`` with columns ``image`` (absolute path) and ``caption``
        (string), index reset.

    Raises:
        FileNotFoundError: If ``annotations_filename`` is missing.
    """
    base_path = Path(base_path)
    annotations_path = base_path / "annotations" / annotations_filename
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"COCO annotations not found at {annotations_path}. "
            f"Run `python -m scripts.prepare_data` to download."
        )

    log.info("loading_coco", path=str(annotations_path))
    with annotations_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    annotations = payload["annotations"]

    img_dir = base_path / images_subdir
    img_cap_pairs = [
        [str(img_dir / (IMAGE_FILENAME_TEMPLATE % sample["image_id"])), sample["caption"]]
        for sample in annotations
    ]
    df = pd.DataFrame(img_cap_pairs, columns=["image", "caption"])

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    df = df.reset_index(drop=True)

    if caption_preprocessor is not None:
        df["caption"] = df["caption"].apply(caption_preprocessor)

    log.info("coco_loaded", rows=len(df), unique_images=df["image"].nunique())
    return df
