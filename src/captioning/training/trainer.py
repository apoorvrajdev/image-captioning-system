"""``Trainer`` — orchestration around ``model.compile + model.fit``.

Wraps notebook cells 22 and 23 in a class so:
    * Tests can construct a Trainer with a tiny dataset and assert
      ``trainer.fit`` returns a sensible history dict.
    * Phase 4 can replace the trainer with a CLI-driven main loop without
      changing the notebook-equivalent behaviour.

The trainer reads the optional training-stability fields off ``TrainConfig``
(``label_smoothing``, ``lr_schedule``, ``warmup_steps``, ...). With defaults
in place every existing config produces a byte-identical compile call to the
notebook; flipping one YAML flag opts a run into the modern recipe without
touching code.
"""

from __future__ import annotations

import json
from pathlib import Path

from captioning.config.schema import AppConfig
from captioning.training.callbacks import default_callbacks
from captioning.training.losses import build_loss
from captioning.training.schedules import build_learning_rate
from captioning.utils.logging import get_logger

log = get_logger(__name__)


def _infer_steps_per_epoch(dataset) -> int | None:
    """Best-effort cardinality probe for a ``tf.data.Dataset``.

    Returns ``None`` when the dataset's cardinality is unknown or infinite.
    Used only to derive ``cosine_decay_steps`` when the user didn't pin it
    explicitly.
    """
    try:
        import tensorflow as tf

        card = int(tf.data.experimental.cardinality(dataset).numpy())
    except Exception:  # — cardinality probing is best-effort
        return None
    if card in (-1, -2):  # UNKNOWN, INFINITE
        return None
    return card


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

    def compile(self, *, steps_per_epoch: int | None = None) -> None:
        """Build optimizer + loss from config and call ``model.compile``.

        Args:
            steps_per_epoch: Used to derive ``cosine_decay_steps`` when the
                config doesn't pin it explicitly. Passing ``None`` falls back
                to the config value (or 1 if neither is set — degenerates to
                immediate floor LR, but still a well-defined schedule).
        """
        import tensorflow as tf

        train = self.config.train
        # Vocab size lives on the decoder's final Dense layer; pulling it here
        # avoids threading the tokenizer through the trainer just for loss.
        vocab_size = int(self.model.decoder.out.units)
        loss = build_loss(train.label_smoothing, vocab_size)

        cosine_steps = train.cosine_decay_steps or (
            (steps_per_epoch or 1) * max(train.epochs - 0, 1)
        )
        learning_rate = build_learning_rate(
            schedule=train.lr_schedule,
            peak_learning_rate=train.learning_rate,
            warmup_steps=train.warmup_steps,
            decay_steps=cosine_steps,
            min_learning_rate=train.min_learning_rate,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
        )
        self._compiled = True
        log.info(
            "model_compiled",
            lr_schedule=train.lr_schedule,
            peak_learning_rate=train.learning_rate,
            warmup_steps=train.warmup_steps,
            cosine_decay_steps=cosine_steps,
            label_smoothing=train.label_smoothing,
        )

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
            self.compile(steps_per_epoch=_infer_steps_per_epoch(train_dataset))

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
