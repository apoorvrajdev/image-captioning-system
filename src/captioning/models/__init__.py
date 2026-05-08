"""Models — Keras layers and the top-level captioning model.

Each layer is in its own file so the architecture reads top-to-bottom in a
file tree, not inside a 200-line cell. Layers compose through ``factory.py``,
which is the single place that wires hyperparameters from ``AppConfig``.

    encoder_cnn.py             InceptionV3 backbone, frozen ImageNet weights
    transformer_encoder.py     1-layer Transformer encoder over image patches
    embeddings.py              Token + positional embeddings
    transformer_decoder.py     Multi-head causal decoder with cross-attention
    captioning_model.py        ``ImageCaptioningModel`` (custom train/test step)
    factory.py                 ``build_caption_model(config, vocab_size)``
"""

from captioning.models.captioning_model import ImageCaptioningModel
from captioning.models.embeddings import Embeddings
from captioning.models.encoder_cnn import build_cnn_encoder
from captioning.models.factory import build_caption_model
from captioning.models.transformer_decoder import TransformerDecoderLayer
from captioning.models.transformer_encoder import TransformerEncoderLayer

__all__ = [
    "Embeddings",
    "ImageCaptioningModel",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "build_caption_model",
    "build_cnn_encoder",
]
