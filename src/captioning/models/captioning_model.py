"""``ImageCaptioningModel`` — top-level Keras model with custom train/test step.

Mirrors notebook cell 20 verbatim by default. The model owns its own loss &
accuracy trackers (rather than using compile-time metrics) because the masked
arithmetic in ``calculate_loss`` / ``calculate_accuracy`` depends on the
caption padding mask, which Keras's standard metric API can't see.

Two opt-in fixes layered on top of the baseline, both controlled by the
constructor (defaults preserve the IEEE notebook quirk exactly):

* ``honour_training_flag_in_test_step``
    The notebook's ``compute_loss_and_acc`` hardcodes ``training=True`` on
    both the encoder and decoder calls, even when invoked from ``test_step``.
    That means dropout is active during validation in the IEEE results.
    Setting this flag to True restores the conventional behaviour — dropout
    off in test_step — so val_loss reflects deployment behaviour and early
    stopping fires on a clean signal.

* ``correct_masked_accuracy``
    The baseline's accuracy tracker is a ``Mean`` of per-batch ratios, which
    weights small batches the same as large ones. Setting this flag to True
    feeds the per-batch token count as ``sample_weight`` so the reported
    metric is a true global token-level masked accuracy.

Both knobs are off by default to keep numeric parity with the notebook; the
trainer flips them on automatically when the user opts in via
``train.honour_training_flag_in_test_step``.
"""

from __future__ import annotations


def _build_captioning_model_class():
    import tensorflow as tf

    class ImageCaptioningModel(tf.keras.Model):
        """Stitches CNN encoder + Transformer encoder + Transformer decoder."""

        def __init__(
            self,
            cnn_model,
            encoder,
            decoder,
            image_aug=None,
            *,
            honour_training_flag_in_test_step: bool = False,
            correct_masked_accuracy: bool = False,
        ) -> None:
            super().__init__()
            self.cnn_model = cnn_model
            self.encoder = encoder
            self.decoder = decoder
            self.image_aug = image_aug
            self.honour_training_flag_in_test_step = honour_training_flag_in_test_step
            self.correct_masked_accuracy = correct_masked_accuracy
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")

        # --- masked metrics (notebook cell 20) -----------------------------

        def calculate_loss(self, y_true, y_pred, mask):
            loss = self.loss(y_true, y_pred)
            mask = tf.cast(mask, dtype=loss.dtype)
            loss *= mask
            return tf.reduce_sum(loss) / tf.reduce_sum(mask)

        def calculate_accuracy(self, y_true, y_pred, mask):
            accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
            accuracy = tf.math.logical_and(mask, accuracy)
            accuracy = tf.cast(accuracy, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

        # --- shared loss/acc step (parity quirk: training=True hardcoded) --

        def compute_loss_and_acc(self, img_embed, captions, training=True):
            # The IEEE notebook hardcoded `training=True` on encoder/decoder
            # calls even from `test_step`, which means dropout is on during
            # validation. Honouring the flag (opt-in) restores the standard
            # behaviour and gives a cleaner val_loss signal.
            effective_training = bool(training) if self.honour_training_flag_in_test_step else True
            encoder_output = self.encoder(img_embed, training=effective_training)
            y_input = captions[:, :-1]
            y_true = captions[:, 1:]
            mask = y_true != 0
            y_pred = self.decoder(y_input, encoder_output, training=effective_training, mask=mask)
            loss = self.calculate_loss(y_true, y_pred, mask)
            acc = self.calculate_accuracy(y_true, y_pred, mask)
            mask_count = tf.reduce_sum(tf.cast(mask, tf.float32))
            return loss, acc, mask_count

        # --- Keras hooks ---------------------------------------------------

        def train_step(self, batch):
            imgs, captions = batch
            if self.image_aug:
                imgs = self.image_aug(imgs)
            img_embed = self.cnn_model(imgs)

            with tf.GradientTape() as tape:
                loss, acc, mask_count = self.compute_loss_and_acc(img_embed, captions)

            train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars, strict=False))
            self.loss_tracker.update_state(loss)
            if self.correct_masked_accuracy:
                # Weight per-batch accuracy by token count so the epoch
                # average is a true global accuracy, not a mean of ratios.
                self.acc_tracker.update_state(acc, sample_weight=mask_count)
            else:
                self.acc_tracker.update_state(acc)

            return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

        def test_step(self, batch):
            imgs, captions = batch
            img_embed = self.cnn_model(imgs)
            loss, acc, mask_count = self.compute_loss_and_acc(img_embed, captions, training=False)
            self.loss_tracker.update_state(loss)
            if self.correct_masked_accuracy:
                self.acc_tracker.update_state(acc, sample_weight=mask_count)
            else:
                self.acc_tracker.update_state(acc)
            return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

        @property
        def metrics(self):
            return [self.loss_tracker, self.acc_tracker]

    return ImageCaptioningModel


ImageCaptioningModel = _build_captioning_model_class()
