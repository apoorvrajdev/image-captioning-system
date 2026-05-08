"""Tests for ``captioning.data.splits.make_image_level_splits``."""

from __future__ import annotations

import pandas as pd

from captioning.data.splits import make_image_level_splits


def _build_corpus(n_images: int = 10, captions_per_image: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_images):
        for j in range(captions_per_image):
            rows.append({"image": f"/img/{i}.jpg", "caption": f"caption {i}-{j}"})
    return pd.DataFrame(rows)


def test_splits_are_image_level() -> None:
    """The same image must NOT appear in both train and val — that's the
    whole point of doing image-level (rather than caption-level) splitting."""
    df = _build_corpus(n_images=10, captions_per_image=5)
    train_imgs, _, val_imgs, _ = make_image_level_splits(df, train_fraction=0.8, seed=0)
    assert set(train_imgs).isdisjoint(set(val_imgs))


def test_splits_preserve_total_count() -> None:
    df = _build_corpus(n_images=10, captions_per_image=5)
    train_imgs, train_caps, val_imgs, val_caps = make_image_level_splits(
        df, train_fraction=0.8, seed=0
    )
    assert len(train_imgs) == len(train_caps)
    assert len(val_imgs) == len(val_caps)
    assert len(train_caps) + len(val_caps) == len(df)


def test_splits_are_seed_reproducible() -> None:
    df = _build_corpus(n_images=20, captions_per_image=3)
    a = make_image_level_splits(df, train_fraction=0.8, seed=123)
    b = make_image_level_splits(df, train_fraction=0.8, seed=123)
    assert a == b


def test_splits_seed_changes_partition() -> None:
    """Different seeds should (almost always) produce different splits."""
    df = _build_corpus(n_images=20, captions_per_image=3)
    a_train, _, _, _ = make_image_level_splits(df, train_fraction=0.8, seed=1)
    b_train, _, _, _ = make_image_level_splits(df, train_fraction=0.8, seed=2)
    assert a_train != b_train


def test_train_fraction_uses_int_truncation_like_notebook() -> None:
    """Notebook cell 11 uses ``int(len(img_keys) * 0.8)``. With 10 images and
    fraction 0.85, that gives 8 train / 2 val. ``round`` would give 9/1.
    Preserve the notebook's int() behaviour."""
    df = _build_corpus(n_images=10, captions_per_image=2)
    train_imgs, _, val_imgs, _ = make_image_level_splits(df, train_fraction=0.85, seed=0)
    train_unique = len(set(train_imgs))
    val_unique = len(set(val_imgs))
    assert train_unique == 8
    assert val_unique == 2
