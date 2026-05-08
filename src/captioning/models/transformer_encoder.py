"""Single-layer Transformer encoder for image patch features.

Mirrors notebook cell 17 verbatim. The encoder is intentionally minimal
(1 attention head, 1 layer, 1 dense projection) because the *image* features
are already produced by InceptionV3 — the Transformer encoder's only job is
to project them into the decoder's embedding dimension and let the decoder
attend across patches.
"""

from __future__ import annotations


def _build_transformer_encoder_class():
    import tensorflow as tf

    class TransformerEncoderLayer(tf.keras.layers.Layer):
        """Norm → Dense → Self-attention → Norm + Add (post-norm wrapper).

        Args:
            embed_dim: Dimensionality fed to the dense projection and used as
                ``key_dim`` for attention. Must equal the decoder's embed_dim.
            num_heads: Attention heads. Notebook uses 1.
        """

        def __init__(self, embed_dim: int, num_heads: int) -> None:
            super().__init__()
            self.layer_norm_1 = tf.keras.layers.LayerNormalization()
            self.layer_norm_2 = tf.keras.layers.LayerNormalization()
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

        def call(self, x, training):
            x = self.layer_norm_1(x)
            x = self.dense(x)
            attn_output = self.attention(
                query=x, value=x, key=x, attention_mask=None, training=training
            )
            return self.layer_norm_2(x + attn_output)

    return TransformerEncoderLayer


TransformerEncoderLayer = _build_transformer_encoder_class()
