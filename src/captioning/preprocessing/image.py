"""Image preprocessing.

Mirrors notebook cell 13 (training pipeline) and cell 25 (inference path).
Both paths must produce *byte-identical* tensors — the model only saw 299x299
images normalised by ``inception_v3.preprocess_input`` during training, so
serving must do exactly that. Centralising the pipeline here is what
eliminates train/serve skew.

The two public functions split responsibilities:
    * ``preprocess_image_tensor`` — operates on an already-decoded image
      tensor. Used by the tf.data pipeline AND inference (after decode).
    * ``load_and_preprocess_image`` — reads bytes from disk, decodes, then
      calls ``preprocess_image_tensor``. Used at inference time.

Both use ``tf.keras.layers.Resizing(299, 299)`` (not ``tf.image.resize``)
because the notebook uses the layer form. ``Resizing`` defaults to bilinear
interpolation and rounds to nearest integer dims, which is the exact behaviour
that produced the IEEE BLEU score.
"""

from __future__ import annotations

INCEPTION_INPUT_SIZE = 299


def preprocess_image_tensor(image: tf.Tensor) -> tf.Tensor:  # type: ignore[name-defined]  # noqa: F821
    """Resize to 299x299 and apply ``inception_v3.preprocess_input``.

    Args:
        image: A 3-D ``tf.Tensor`` of shape ``[H, W, 3]`` and dtype ``uint8``
            or ``float32``. The Resizing layer accepts both.

    Returns:
        ``tf.Tensor`` of shape ``[299, 299, 3]``, dtype ``float32``, with the
        InceptionV3 normalisation applied (pixel values in ``[-1, 1]``).
    """
    import tensorflow as tf

    image = tf.keras.layers.Resizing(INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE)(image)
    return tf.keras.applications.inception_v3.preprocess_input(image)


def load_and_preprocess_image(image_path: str) -> tf.Tensor:  # type: ignore[name-defined]  # noqa: F821
    """Read a JPEG from disk and run it through ``preprocess_image_tensor``.

    Args:
        image_path: Path to a JPEG file. Strings, ``pathlib.Path``, and
            ``tf.string`` tensors all work — the latter matters because
            ``tf.data`` pipelines pass paths as tensors.

    Returns:
        A 3-D ``tf.Tensor`` ready to feed into the CNN encoder.

    Raises:
        tf.errors.NotFoundError: If the file does not exist.
        tf.errors.InvalidArgumentError: If the file is not a valid JPEG/PNG.
    """
    import tensorflow as tf

    raw = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(raw, channels=3)
    return preprocess_image_tensor(image)
