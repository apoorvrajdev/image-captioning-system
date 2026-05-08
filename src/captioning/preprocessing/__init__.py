"""Preprocessing — pure transforms on captions and images.

Functions in this package take inputs and return outputs with no hidden state
and no disk I/O. That makes them trivially unit-testable and lets us share the
same logic across the training pipeline (where they're composed into tf.data
maps) and the inference path (where they're called once per request).

Modules:
    caption.py        ``preprocess_caption(text)`` — lower/strip/wrap with [start]/[end]
    image.py          ``preprocess_image_tensor(img)``, ``load_and_preprocess_image(path)``
    tokenizer.py      ``CaptionTokenizer`` — wraps tf.keras TextVectorization
    augmentation.py   ``default_image_augmentation()`` — Keras Sequential
"""

from captioning.preprocessing.augmentation import default_image_augmentation
from captioning.preprocessing.caption import (
    END_TOKEN,
    START_TOKEN,
    preprocess_caption,
)
from captioning.preprocessing.image import (
    load_and_preprocess_image,
    preprocess_image_tensor,
)
from captioning.preprocessing.tokenizer import CaptionTokenizer

__all__ = [
    "END_TOKEN",
    "START_TOKEN",
    "CaptionTokenizer",
    "default_image_augmentation",
    "load_and_preprocess_image",
    "preprocess_caption",
    "preprocess_image_tensor",
]
