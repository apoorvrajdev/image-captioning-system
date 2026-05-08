"""``Trainer`` — orchestration around ``model.compile + model.fit``.

Wraps notebook cells 22 and 23 in a class so:
    * Tests can construct a Trainer with a tiny dataset and assert
      ``trainer.fit`` returns a sensible history dict.
    * Phase 4 can replace the trainer with a CLI-driven main loop without
      changing the notebook-equivalent behaviour.

The trainer is intentionally thin — no MLflow integration yet (Phase 2
adds it), no distributed strategy (out of scope for the IEEE notebook).
"""

from __future__ import annotations

import json
from pathlib import Path

from captioning.config.schema import AppConfig
from captioning.training.callbacks import default_callbacks
from captioning.training.losses import masked_sparse_categorical_crossentropy
from captioning.utils.logging import get_logger

log = get_logger(__name__)


class Trainer:
    """Thin orchestration layer around an ``ImageCaptioningModel``."""

    def __init__(self, model, config: AppConfig) -> None:
        """Args:
        model: Result of ``build_caption_model(config, vocab_size)``.
        config: Validated ``AppConfig``.
        """
        self.model = model
        self.config = config
        self._compiled = False

    def compile(self) -> None:
        """Apply the same ``compile`` call the notebook makes (cell 22)."""
        import tensorflow as tf

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.train.learning_rate),
            loss=masked_sparse_categorical_crossentropy(),
        )
        self._compiled = True
        log.info("model_compiled", learning_rate=self.config.train.learning_rate)

    def fit(
        self,
        train_dataset,
        val_dataset,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run ``model.fit`` and return a history dict.

        Args:
            train_dataset: ``tf.data.Dataset`` from
                ``data.pipeline.build_train_pipeline``.
            val_dataset: ``tf.data.Dataset`` from
                ``data.pipeline.build_val_pipeline``.
            output_dir: If provided, callbacks write ``best.h5`` and
                ``training_log.csv`` here, and ``history.json`` is dumped at
                the end.

        Returns:
            ``history.history`` as a ``dict[str, list[float]]``.
        """
        if not self._compiled:
            self.compile()

        callbacks = default_callbacks(self.config, output_dir=output_dir)
        log.info("fit_start", epochs=self.config.train.epochs)
        history = self.model.fit(
            train_dataset,
            epochs=self.config.train.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
        )
        log.info("fit_end", final_loss=history.history.get("loss", [None])[-1])

        if output_dir is not None:
            history_path = Path(output_dir) / "history.json"
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(history.history, f, indent=2)

        return dict(history.history)
