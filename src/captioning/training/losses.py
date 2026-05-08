"""Training losses.

The notebook (cell 22) compiles the model with::

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

Why ``reduction="none"``: the model's ``calculate_loss`` (cell 20) does the
reduction itself, multiplying by the padding mask before averaging. A built-in
reduction would average over the padded tokens too, biasing the loss.

We expose the loss via a tiny factory rather than a constant so callers don't
have to import TF themselves to get it.
"""

from __future__ import annotations


def masked_sparse_categorical_crossentropy():
    """Return the loss function the model is compiled with.

    Same as notebook cell 22: ``from_logits=False, reduction="none"``. The
    decoder applies a softmax already (``Dense(..., activation="softmax")``)
    so logits=False is correct.
    """
    import tensorflow as tf

    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
