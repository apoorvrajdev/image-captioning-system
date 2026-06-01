"""``tf.data`` pipeline construction.

Mirrors notebook cell 13 (the ``load_data`` map function) and cell 14 (the
two pipeline definitions). The notebook closes over a global ``tokenizer``;
we pass the tokenizer in explicitly so the same code works in tests, scripts,
and the parity audit.

Note on ``shuffle`` in the val pipeline: the notebook shuffles both train
and val. That is technically unnecessary for validation but harmless, and we
preserve it for parity. Phase 1b removes it from val.
"""

from __future__ import annotations

from collections.abc import Sequence

from captioning.preprocessing.image import preprocess_image_tensor
from captioning.preprocessing.tokenizer import CaptionTokenizer


def _make_load_data_fn(tokenizer: CaptionTokenizer):
    """Return a ``tf.data``-compatible map function (image_path, caption) -> (image, ids).

    Defined as a closure rather than a top-level function so it captures the
    tokenizer without leaking it into the module namespace. ``tf.data`` calls
    this for every example with both arguments as ``tf.string`` tensors.
    """
    import tensorflow as tf

    def load_data(image_path, caption):
        raw = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = preprocess_image_tensor(image)
        ids = tokenizer.encode(caption)
        return image, ids

    return load_data


def build_train_pipeline(
    image_paths: Sequence[str],
    captions: Sequence[str],
    tokenizer: CaptionTokenizer,
    batch_size: int,
    buffer_size: int,
):
    """Build the training dataset, byte-identical to notebook cell 14.

    Args:
        image_paths: One path per (image, caption) pair (image-level split
            already applied — see ``data.splits``).
        captions: Preprocessed captions, one per ``image_paths`` entry.
        tokenizer: Fitted ``CaptionTokenizer``.
        batch_size: Mini-batch size; matches ``BATCH_SIZE`` in the notebook.
        buffer_size: Shuffle buffer size; matches ``BUFFER_SIZE``.

    Returns:
        A ``tf.data.Dataset`` yielding ``(image, token_ids)`` batches.
    """
    import tensorflow as tf

    load_data = _make_load_data_fn(tokenizer)
    return (
        tf.data.Dataset.from_tensor_slices((list(image_paths), list(captions)))
        .map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size)
        .batch(batch_size)
    )


def build_val_pipeline(
    image_paths: Sequence[str],
    captions: Sequence[str],
    tokenizer: CaptionTokenizer,
    batch_size: int,
    buffer_size: int,
):
    """Build the validation dataset.

    Identical structure to ``build_train_pipeline``, with a separate function
    so Phase 1b can drop the (unnecessary) shuffle from val without coupling
    the change to train.
    """
    import tensorflow as tf

    load_data = _make_load_data_fn(tokenizer)
    return (
        tf.data.Dataset.from_tensor_slices((list(image_paths), list(captions)))
        .map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size)  # Notebook cell 14 shuffles val too — preserved.
        .batch(batch_size)
    )
