"""Learning-rate schedules.

The baseline pipeline uses a constant Adam LR (matching the IEEE notebook),
which is fine for short fine-tuning runs but tends to leave Transformer
captioners in a mediocre local minimum: the LR is too aggressive at start
(decoder weights are still random) and too high near convergence (the model
oscillates around a flat basin instead of settling).

The fix the literature converged on is linear warmup followed by cosine
decay (the GPT/BERT/ViT recipe):

    lr(step) = peak_lr * step / warmup_steps                if step < warmup
    lr(step) = min_lr + (peak_lr - min_lr) * 0.5 *
               (1 + cos(pi * (step - warmup) / decay_steps)) otherwise

We implement it as a ``LearningRateSchedule`` so the optimizer can call it
per-step automatically, without us having to track step counts manually.
"""

from __future__ import annotations


def _build_warmup_cosine_class():
    """Lazy-build the schedule class to keep TF off the package import path."""
    import tensorflow as tf

    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        """Linear warmup followed by cosine decay to ``min_learning_rate``.

        Args:
            peak_learning_rate: Maximum LR reached at the end of warmup.
            warmup_steps: Number of steps to linearly ramp from 0 to peak.
                ``0`` disables warmup (starts directly at ``peak``).
            decay_steps: Number of steps over which to cosine-decay from
                ``peak`` to ``min_learning_rate`` after warmup.
            min_learning_rate: Floor reached at the end of decay (and held
                thereafter).
        """

        def __init__(
            self,
            peak_learning_rate: float,
            warmup_steps: int,
            decay_steps: int,
            min_learning_rate: float = 0.0,
        ) -> None:
            super().__init__()
            self.peak_learning_rate = float(peak_learning_rate)
            self.warmup_steps = int(warmup_steps)
            self.decay_steps = max(int(decay_steps), 1)
            self.min_learning_rate = float(min_learning_rate)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            peak = tf.constant(self.peak_learning_rate, dtype=tf.float32)
            floor = tf.constant(self.min_learning_rate, dtype=tf.float32)
            warmup = tf.constant(float(self.warmup_steps), dtype=tf.float32)
            decay = tf.constant(float(self.decay_steps), dtype=tf.float32)

            # During warmup: linear ramp 0 -> peak.
            warmup_lr = peak * tf.math.divide_no_nan(step, warmup)

            # After warmup: cosine decay peak -> floor over decay_steps.
            progress = tf.minimum(1.0, tf.math.divide_no_nan(step - warmup, decay))
            cosine = 0.5 * (1.0 + tf.cos(tf.constant(3.141592653589793) * progress))
            decay_lr = floor + (peak - floor) * cosine

            return tf.where(step < warmup, warmup_lr, decay_lr)

        def get_config(self) -> dict[str, float | int]:
            return {
                "peak_learning_rate": self.peak_learning_rate,
                "warmup_steps": self.warmup_steps,
                "decay_steps": self.decay_steps,
                "min_learning_rate": self.min_learning_rate,
            }

    return WarmupCosineDecay


WarmupCosineDecay = _build_warmup_cosine_class()


def build_learning_rate(
    *,
    schedule: str,
    peak_learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
    min_learning_rate: float,
):
    """Return either a float (constant LR) or a :class:`WarmupCosineDecay`.

    The optimizer treats a float as a fixed LR and a ``LearningRateSchedule``
    as a per-step callable — we hide that asymmetry behind this factory so
    the trainer only ever passes ``learning_rate=build_learning_rate(...)``.
    """
    if schedule == "constant":
        return peak_learning_rate
    if schedule == "cosine":
        return WarmupCosineDecay(
            peak_learning_rate=peak_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_learning_rate=min_learning_rate,
        )
    raise ValueError(f"unsupported lr_schedule: {schedule!r}")
