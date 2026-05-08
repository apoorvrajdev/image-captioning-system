"""Inference-time image loader — same path as cell 25 of the notebook.

The training pipeline goes through ``data.pipeline.build_*_pipeline`` which
calls ``preprocessing.image.preprocess_image_tensor``. The inference path
must produce the same tensor for the same image, otherwise BLEU drops
silently. This module re-uses ``preprocess_image_tensor`` so train/serve
parity is by construction.
"""

from __future__ import annotations

from captioning.preprocessing.image import preprocess_image_tensor


def load_image_from_path(image_path: str):
    """Read a JPEG/PNG from disk and produce a model-ready tensor.

    Mirrors the ``load_image_from_path`` helper in notebook cell 25.

    Args:
        image_path: Filesystem path to the image. ``str``, ``Path``, and
            ``tf.string`` tensors all work (TF does the conversion).

    Returns:
        A ``tf.Tensor`` of shape ``[299, 299, 3]``, dtype ``float32``,
        with InceptionV3 normalisation.
    """
    import tensorflow as tf

    raw = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(raw, channels=3)
    return preprocess_image_tensor(image)
