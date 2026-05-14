"""Training — losses, schedules, callbacks, and the trainer.

The notebook computes loss + masked accuracy inside the model's ``train_step``;
we keep that structure for parity but expose the loss function and callbacks
as standalone modules so they can be unit-tested and reused.

    losses.py      ``masked_sparse_categorical_crossentropy`` (baseline) +
                   ``label_smoothed_crossentropy`` + ``build_loss``
    schedules.py   ``WarmupCosineDecay`` + ``build_learning_rate``
    callbacks.py   ``default_callbacks(config)`` — early stopping + checkpoint
    trainer.py     ``Trainer.fit()`` — wraps compile + fit + history serialization
"""

from captioning.training.callbacks import default_callbacks
from captioning.training.losses import (
    build_loss,
    label_smoothed_crossentropy,
    masked_sparse_categorical_crossentropy,
)
from captioning.training.schedules import WarmupCosineDecay, build_learning_rate
from captioning.training.trainer import Trainer

__all__ = [
    "Trainer",
    "WarmupCosineDecay",
    "build_learning_rate",
    "build_loss",
    "default_callbacks",
    "label_smoothed_crossentropy",
    "masked_sparse_categorical_crossentropy",
]
