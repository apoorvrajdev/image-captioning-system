"""Multi-head Transformer decoder with causal masking and cross-attention.

Mirrors notebook cell 19. Two changes from the notebook, both behaviour-
preserving when defaults match:

1. **Globals are now constructor arguments.** The notebook closes over
   ``tokenizer.vocabulary_size()`` and ``MAX_LENGTH`` from module scope.
   We pass them in as ``vocab_size`` and ``max_len`` so the decoder can be
   instantiated in tests, factories, and notebooks without setting up a
   global tokenizer first.
2. **Dropout rates and attention head count are configurable** with the
   notebook values as defaults. This costs nothing today and lets Phase 1b
   ablations vary them without code changes.
"""

from __future__ import annotations

from captioning.models.embeddings import Embeddings


def _build_transformer_decoder_class():
    import tensorflow as tf

    class TransformerDecoderLayer(tf.keras.layers.Layer):
        """Causal self-attention + cross-attention + FFN block.

        Args:
            embed_dim: Token/positional embedding dimension. Must equal the
                encoder's ``embed_dim``.
            units: Hidden dimension of the feed-forward sub-block.
            num_heads: Multi-head attention heads. Notebook uses 8.
            vocab_size: Output projection dimension (the model emits softmax
                probabilities over the vocabulary).
            max_len: Maximum decode length, used to size positional embeddings.
            attention_dropout: Dropout applied inside MultiHeadAttention.
                Notebook uses 0.1.
            inner_dropout: Dropout after the first dense layer in the FFN.
                Notebook uses 0.3.
            outer_dropout: Dropout after the residual + final layernorm.
                Notebook uses 0.5.
        """

        def __init__(
            self,
            embed_dim: int,
            units: int,
            num_heads: int,
            vocab_size: int,
            max_len: int,
            attention_dropout: float = 0.1,
            inner_dropout: float = 0.3,
            outer_dropout: float = 0.5,
        ) -> None:
            super().__init__()
            self.embedding = Embeddings(vocab_size, embed_dim, max_len)

            self.attention_1 = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim, dropout=attention_dropout
            )
            self.attention_2 = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim, dropout=attention_dropout
            )

            self.layernorm_1 = tf.keras.layers.LayerNormalization()
            self.layernorm_2 = tf.keras.layers.LayerNormalization()
            self.layernorm_3 = tf.keras.layers.LayerNormalization()

            self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
            self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

            self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")

            self.dropout_1 = tf.keras.layers.Dropout(inner_dropout)
            self.dropout_2 = tf.keras.layers.Dropout(outer_dropout)

        def call(self, input_ids, encoder_output, training, mask=None):
            embeddings = self.embedding(input_ids)

            combined_mask = None
            padding_mask = None

            if mask is not None:
                causal_mask = self.get_causal_attention_mask(embeddings)
                padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
                combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
                combined_mask = tf.minimum(combined_mask, causal_mask)

            attn_output_1 = self.attention_1(
                query=embeddings,
                value=embeddings,
                key=embeddings,
                attention_mask=combined_mask,
                training=training,
            )
            out_1 = self.layernorm_1(embeddings + attn_output_1)

            attn_output_2 = self.attention_2(
                query=out_1,
                value=encoder_output,
                key=encoder_output,
                attention_mask=padding_mask,
                training=training,
            )
            out_2 = self.layernorm_2(out_1 + attn_output_2)

            ffn_out = self.ffn_layer_1(out_2)
            ffn_out = self.dropout_1(ffn_out, training=training)
            ffn_out = self.ffn_layer_2(ffn_out)

            ffn_out = self.layernorm_3(ffn_out + out_2)
            ffn_out = self.dropout_2(ffn_out, training=training)
            return self.out(ffn_out)

        def get_causal_attention_mask(self, inputs):
            input_shape = tf.shape(inputs)
            batch_size, sequence_length = input_shape[0], input_shape[1]
            i = tf.range(sequence_length)[:, tf.newaxis]
            j = tf.range(sequence_length)
            mask = tf.cast(i >= j, dtype="int32")
            mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
            mult = tf.concat(
                [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
                axis=0,
            )
            return tf.tile(mask, mult)

    return TransformerDecoderLayer


TransformerDecoderLayer = _build_transformer_decoder_class()
