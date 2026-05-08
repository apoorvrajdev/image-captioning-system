"""Token + positional embedding layer.

Mirrors notebook cell 18 verbatim. The decoder learns its own positional
encoding (rather than using sinusoidal) — that's the published architecture,
preserved here.
"""

from __future__ import annotations


def _import_tf():
    """Local import keeps top-level package import lightweight.

    Without this, ``from captioning.models import Embeddings`` would trigger
    a multi-second TF import even for callers that don't use it.
    """
    import tensorflow as tf

    return tf


# Defining the class lazily inside a factory keeps TF out of the import path.
# Callers do ``Embeddings = _build_embeddings_class()`` once at module init.
def _build_embeddings_class():
    tf = _import_tf()

    class Embeddings(tf.keras.layers.Layer):
        """Sum of token and learned positional embeddings.

        Args:
            vocab_size: Size of the token vocabulary
                (``CaptionTokenizer.vocabulary_size``).
            embed_dim: Dimensionality of each embedding vector
                (``model.embedding_dim``, default 512).
            max_len: Maximum sequence length (``model.max_length``, default 40).
        """

        def __init__(self, vocab_size: int, embed_dim: int, max_len: int) -> None:
            super().__init__()
            self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
            self.position_embeddings = tf.keras.layers.Embedding(
                max_len, embed_dim, input_shape=(None, max_len)
            )

        def call(self, input_ids):
            length = tf.shape(input_ids)[-1]
            position_ids = tf.range(start=0, limit=length, delta=1)
            position_ids = tf.expand_dims(position_ids, axis=0)
            token_embeddings = self.token_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            return token_embeddings + position_embeddings

    return Embeddings


Embeddings = _build_embeddings_class()
