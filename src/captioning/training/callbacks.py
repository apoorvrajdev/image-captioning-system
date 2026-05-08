"""Default training callbacks.

Mirrors notebook cell 22 (``EarlyStopping(patience=3, restore_best_weights=True)``)
and adds Phase-2 hooks (``ModelCheckpoint``, ``CSVLogger``) that the trainer
will use. Each callback is created by a tiny factory so callers don't have to
import TF for the names.
"""

from __future__ import annotations

from pathlib import Path

from captioning.config.schema import AppConfig


def default_callbacks(
    config: AppConfig,
    *,
    output_dir: str | Path | None = None,
):
    """Return the list of callbacks ``Trainer.fit`` will pass to ``model.fit``.

    Args:
        config: App config (uses ``train.early_stopping_patience``).
        output_dir: If provided, ``ModelCheckpoint`` writes ``best.h5`` and
            ``CSVLogger`` writes ``training_log.csv`` here. Notebook does
            neither — these are Phase-1b improvements layered on top of the
            parity baseline. They run *before* parity is exercised because
            adding a callback does not change loss values, only emits files.

    Returns:
        A list of ``tf.keras.callbacks.Callback`` instances.
    """
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.train.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        callbacks += [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(out / "best.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
            ),
            tf.keras.callbacks.CSVLogger(str(out / "training_log.csv")),
        ]
    return callbacks
