"""Image-augmentation pipeline (training only).

Mirrors notebook cell 15. Augmentation is deliberately separate from
``image.py``: augmentations introduce randomness and only run during training,
while ``preprocess_image_tensor`` is deterministic and runs in both train and
serve. Mixing them risks accidentally augmenting at inference time.
"""

from __future__ import annotations


def default_image_augmentation() -> tf.keras.Sequential:  # type: ignore[name-defined]  # noqa: F821
    """Build the augmentation chain used during training.

    The model is composed once (notebook cell 21::

        ImageCaptioningModel(..., image_aug=image_augmentation)

    ) and the augmentation block runs only inside ``train_step`` (notebook
    cell 20). ``test_step`` skips augmentation, which is the correct behaviour
    we preserve.

    Returns:
        A ``tf.keras.Sequential`` of ``RandomFlip`` + ``RandomRotation`` +
        ``RandomContrast`` matching cell 15 exactly.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.3),
        ]
    )
