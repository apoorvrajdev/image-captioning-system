"""Training losses.

The notebook (cell 22) compiles the model with::

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

Why ``reduction="none"``: the model's ``calculate_loss`` (cell 20) does the
reduction itself, multiplying by the padding mask before averaging. A built-in
reduction would average over the padded tokens too, biasing the loss.

For the stabilisation phase we also support label-smoothed cross-entropy.
Label smoothing replaces the one-hot target ``y_true`` with a mixture of the
true label and a uniform distribution over the vocabulary:

    target = (1 - eps) * one_hot(y) + eps / vocab_size

The decoder's output is already softmaxed (`Dense(..., activation='softmax')`),
so the loss reduces to ``-sum(target * log(p), axis=-1)``. Smoothing
discourages the decoder from collapsing to a few high-probability tokens —
the most common failure mode of cross-entropy-trained captioners and a
likely root cause of the generic captions we're trying to fix.
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


def label_smoothed_crossentropy(label_smoothing: float, vocab_size: int):
    """Per-token cross-entropy with uniform label smoothing.

    Returned callable has the same signature as the sparse loss above
    (``loss(y_true, y_pred) -> [B, T]``) so the model's masking machinery in
    ``ImageCaptioningModel.calculate_loss`` works unchanged.

    Args:
        label_smoothing: Smoothing strength in ``[0, 1)``. ``0.0`` reduces to
            the sparse-categorical baseline.
        vocab_size: Size of the output distribution (matches the decoder's
            final ``Dense`` units). Used to compute the uniform component.
    """
    import tensorflow as tf

    if label_smoothing == 0.0:
        return masked_sparse_categorical_crossentropy()

    eps = float(label_smoothing)
    log_eps = tf.constant(1e-12, dtype=tf.float32)
    vocab = int(vocab_size)
    uniform = eps / float(vocab)

    def loss_fn(y_true, y_pred):
        # y_true: [B, T] int ids; y_pred: [B, T, V] softmax probabilities.
        y_pred = tf.cast(y_pred, tf.float32)
        one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=vocab, dtype=tf.float32)
        target = one_hot * (1.0 - eps) + uniform
        # Standard cross-entropy on softmax probs. Add log_eps to avoid log(0)
        # on padding columns where the model would otherwise emit 0.
        return -tf.reduce_sum(target * tf.math.log(y_pred + log_eps), axis=-1)

    return loss_fn


def build_loss(label_smoothing: float, vocab_size: int):
    """Pick the right loss based on ``label_smoothing``.

    Convenience wrapper so the trainer never has to branch on the smoothing
    value itself — it always calls ``build_loss(...)`` and the right
    implementation comes back.
    """
    if label_smoothing == 0.0:
        return masked_sparse_categorical_crossentropy()
    return label_smoothed_crossentropy(label_smoothing, vocab_size)
