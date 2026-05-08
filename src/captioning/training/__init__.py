"""Training — losses, callbacks, and the trainer that orchestrates ``model.fit``.

The notebook computes loss + masked accuracy inside the model's ``train_step``;
we keep that structure for parity but expose the loss function and callbacks
as standalone modules so they can be unit-tested and reused (e.g. by Phase 1b
beam-search evaluators).

    losses.py      ``masked_sparse_categorical_crossentropy`` — the same loss the notebook uses
    callbacks.py   ``default_callbacks(config)`` — early stopping (and Phase 4 checkpoint hooks)
    trainer.py     ``Trainer.fit()`` — wraps compile + fit + history serialization
"""

from captioning.training.callbacks import default_callbacks
from captioning.training.losses import masked_sparse_categorical_crossentropy
from captioning.training.trainer import Trainer

__all__ = [
    "Trainer",
    "default_callbacks",
    "masked_sparse_categorical_crossentropy",
]
