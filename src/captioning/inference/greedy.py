"""Greedy caption generation.

Mirrors notebook cell 25's ``generate_caption`` exactly. The notebook closes
over four globals (``caption_model``, ``tokenizer``, ``idx2word``,
``MAX_LENGTH``); we accept them as explicit arguments so the function is
callable from tests, scripts, FastAPI, and the parity audit.

The algorithm:
    1. CNN-encode the image.
    2. Transformer-encode the patch features.
    3. Seed the caption with ``[start]``.
    4. For each position 0 ... ``max_length - 2``:
        a. Tokenise the partial caption (``[:, :-1]`` because TextVectorization
           pads to ``max_length`` and we feed ``max_length - 1`` positions
           into the decoder).
        b. Decode and take the argmax at the current position.
        c. Stop on ``[end]``; otherwise append the predicted word.
    5. Strip the ``[start]`` prefix and return.
"""

from __future__ import annotations

from captioning.preprocessing.caption import END_TOKEN, START_TOKEN
from captioning.preprocessing.tokenizer import CaptionTokenizer


def generate_caption_greedy(
    model,
    tokenizer: CaptionTokenizer,
    image_tensor,
    max_length: int,
    *,
    add_noise: bool = False,
) -> str:
    """Generate a caption for one image using greedy (argmax) decoding.

    Args:
        model: An ``ImageCaptioningModel`` whose weights have been loaded.
        tokenizer: Fitted ``CaptionTokenizer`` (the same one used at training).
        image_tensor: A ``[299, 299, 3]`` float tensor produced by
            ``inference.load_image_from_path`` (or ``preprocess_image_tensor``).
        max_length: Decode budget — equals ``config.model.max_length`` (40
            in the notebook).
        add_noise: Replicates the notebook's ``add_noise`` knob; off by default.

    Returns:
        The generated caption string with the ``[start]`` sentinel removed.
        The ``[end]`` sentinel is naturally absent because the loop breaks on it.
    """
    import numpy as np
    import tensorflow as tf

    img = image_tensor
    if add_noise:
        noise = tf.random.normal(img.shape) * 0.1
        img = img + noise
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))

    img = tf.expand_dims(img, axis=0)
    img_embed = model.cnn_model(img)
    img_encoded = model.encoder(img_embed, training=False)

    y_inp = START_TOKEN
    for i in range(max_length - 1):
        tokenized = tokenizer.encode([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = model.decoder(tokenized, img_encoded, training=False, mask=mask)

        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = tokenizer.decode_id(pred_idx)
        if pred_word == END_TOKEN:
            break
        y_inp += " " + pred_word

    return y_inp.replace(f"{START_TOKEN} ", "")
