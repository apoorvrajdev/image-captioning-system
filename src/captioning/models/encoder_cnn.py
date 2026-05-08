"""InceptionV3 image encoder.

Mirrors notebook cell 16. The encoder is the *frozen* visual backbone that
turns a 299x299 RGB image into a sequence of 2048-dimensional feature vectors
(one per spatial position in InceptionV3's last conv layer). The Transformer
encoder/decoder learn on top of these features; the InceptionV3 weights are
never updated during training.

Why a build function and not a Keras layer? The CNN is constructed from a
pretrained model whose weights are downloaded the first time. Wrapping
construction in a function gives callers a single line to invoke, and lets
us add caching / offline-loading paths later without touching call sites.
"""

from __future__ import annotations


def build_cnn_encoder():
    """Build the InceptionV3 backbone with the classification head removed.

    Returns:
        A ``tf.keras.Model`` mapping ``[B, 299, 299, 3]`` images to
        ``[B, 64, 2048]`` patch features (8x8=64 spatial positions, each a
        2048-dim vector — InceptionV3's ``mixed10`` layer).
    """
    import tensorflow as tf

    inception = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
    )

    output = inception.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)

    return tf.keras.models.Model(inception.input, output)
