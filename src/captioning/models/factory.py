"""``build_caption_model(config, vocab_size)`` — single place to wire layers.

Mirrors notebook cell 21::

    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)
    cnn_model = CNN_Encoder()
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder,
        image_aug=image_augmentation,
    )

Pulling this into a factory function isolates "how layers are wired" from
"what hyperparameters they use", so Phase 1b ablations and Phase 5 model
swaps only touch this file.
"""

from __future__ import annotations

from captioning.config.schema import AppConfig
from captioning.models.captioning_model import ImageCaptioningModel
from captioning.models.encoder_cnn import build_cnn_encoder
from captioning.models.transformer_decoder import TransformerDecoderLayer
from captioning.models.transformer_encoder import TransformerEncoderLayer
from captioning.preprocessing.augmentation import default_image_augmentation


def build_caption_model(
    config: AppConfig,
    vocab_size: int,
    *,
    use_augmentation: bool = True,
):
    """Construct a ready-to-compile ``ImageCaptioningModel``.

    Args:
        config: Validated app config (the ``model`` section is consumed here).
        vocab_size: Comes from the *fitted* tokenizer
            (``CaptionTokenizer.vocabulary_size``). The factory does not own
            tokenizer state — callers fit the tokenizer first, pass the size in.
        use_augmentation: If True (default), wires
            ``default_image_augmentation()`` for ``train_step``. Inference and
            evaluation paths pass False.

    Returns:
        An uncompiled ``ImageCaptioningModel``. Caller is responsible for
        ``model.compile(optimizer=..., loss=...)``.
    """
    m = config.model

    encoder = TransformerEncoderLayer(m.embedding_dim, m.encoder_num_heads)
    decoder = TransformerDecoderLayer(
        embed_dim=m.embedding_dim,
        units=m.units,
        num_heads=m.decoder_num_heads,
        vocab_size=vocab_size,
        max_len=m.max_length,
        attention_dropout=m.decoder_attention_dropout,
        inner_dropout=m.decoder_dropout_inner,
        outer_dropout=m.decoder_dropout_outer,
    )
    cnn = build_cnn_encoder()
    aug = default_image_augmentation() if use_augmentation else None
    return ImageCaptioningModel(cnn_model=cnn, encoder=encoder, decoder=decoder, image_aug=aug)
