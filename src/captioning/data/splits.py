"""Image-level train/val splitting.

Mirrors notebook cell 11. The split is intentionally at the *image* level
(not the caption level): every image owns ~5 captions in COCO, and putting
some of an image's captions in train and others in val would be data leakage.

The notebook does this correctly via the ``img_to_cap_vector`` defaultdict
loop; we preserve that exact algorithm but inject the seed so the split is
reproducible across runs.
"""

from __future__ import annotations

import collections
import random

import pandas as pd

from captioning.utils.logging import get_logger

log = get_logger(__name__)


def make_image_level_splits(
    captions: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split captions into train/val while keeping all of an image's
    captions in the same split.

    Mirrors notebook cell 11 exactly when ``seed`` is the same value that was
    fed to ``random.seed`` before the notebook ran. ``seed=None`` reproduces
    the notebook's non-deterministic behaviour.

    Args:
        captions: DataFrame with ``image`` and ``caption`` columns
            (preprocessed if you want preprocessed splits — the loader applies
            ``preprocess_caption`` upstream).
        train_fraction: Fraction of *unique images* assigned to the train
            split. The notebook uses ``int(len(img_keys) * 0.8)``, which we
            preserve byte-for-byte (``int()`` truncates, not rounds).
        seed: If provided, used to seed Python's ``random`` for the shuffle.

    Returns:
        Tuple ``(train_imgs, train_captions, val_imgs, val_captions)`` where
        each list has one entry per (image, caption) pair, expanded so an
        image with N captions appears N times.
    """
    img_to_cap = collections.defaultdict(list)
    for img, cap in zip(captions["image"], captions["caption"], strict=True):
        img_to_cap[img].append(cap)

    img_keys = list(img_to_cap.keys())

    if seed is not None:
        rng = random.Random(seed)  # — seeded RNG is reproducible by design
        rng.shuffle(img_keys)
    else:
        random.shuffle(img_keys)

    slice_index = int(len(img_keys) * train_fraction)
    train_keys, val_keys = img_keys[:slice_index], img_keys[slice_index:]

    train_imgs: list[str] = []
    train_captions: list[str] = []
    for k in train_keys:
        n = len(img_to_cap[k])
        train_imgs.extend([k] * n)
        train_captions.extend(img_to_cap[k])

    val_imgs: list[str] = []
    val_captions: list[str] = []
    for k in val_keys:
        n = len(img_to_cap[k])
        val_imgs.extend([k] * n)
        val_captions.extend(img_to_cap[k])

    log.info(
        "splits_made",
        train_images=len(train_keys),
        val_images=len(val_keys),
        train_captions=len(train_captions),
        val_captions=len(val_captions),
    )
    return train_imgs, train_captions, val_imgs, val_captions
