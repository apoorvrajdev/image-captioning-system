"""Data — dataset loaders, splits, and tf.data pipelines.

Data lives separately from preprocessing because *I/O and randomness* are
fundamentally different concerns from pure transforms. This package owns:

    coco.py       Read COCO annotation JSONs into a (image_path, caption) DataFrame
    splits.py     Deterministic image-level train/val splitting (NOT caption-level —
                  preventing the same image from appearing in both splits)
    pipeline.py   Compose preprocessing + tokenization into tf.data pipelines
"""

from captioning.data.coco import load_coco_annotations
from captioning.data.pipeline import build_train_pipeline, build_val_pipeline
from captioning.data.splits import make_image_level_splits

__all__ = [
    "build_train_pipeline",
    "build_val_pipeline",
    "load_coco_annotations",
    "make_image_level_splits",
]
