"""``CaptionTokenizer`` — typed wrapper around ``tf.keras.layers.TextVectorization``.

Why a wrapper instead of using the Keras layer directly?

1. **Stable interface for the model.** The model code calls
   ``tokenizer.encode(captions)`` and ``tokenizer.decode_id(idx)``. The fact
   that those happen to delegate to a Keras layer is an implementation
   detail. In Phase 5 we may swap the implementation for HuggingFace
   ``tokenizers`` without rewriting the encoder, decoder, or inference loop.
2. **Persistence.** The notebook saves the *vocabulary list* with pickle, but
   loading requires re-instantiating a layer and calling ``set_vocabulary``.
   That ceremony belongs inside the wrapper, not at every call site.
3. **A JSON sidecar.** Pickle is fast but opaque and risky to load from
   untrusted sources. We additionally write a ``vocab.json`` file (one token
   per line, UTF-8) so humans and other tools can inspect the vocabulary.

The wrapper preserves the notebook's behaviour exactly: ``standardize=None``,
``output_sequence_length`` defaults to ``max_length``, and ``encode`` accepts
either a single string or a list of strings (matching the layer's call form
used in cells 7 and 25).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path

VOCAB_PICKLE_FILENAME = "vocab.pkl"
VOCAB_JSON_FILENAME = "vocab.json"


class CaptionTokenizer:
    """Wrapper that owns a fitted ``TextVectorization`` layer + lookup tables."""

    def __init__(self, vocab_size: int, max_length: int) -> None:
        """Construct an unfit tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (notebook: ``VOCABULARY_SIZE``).
            max_length: Pad/truncate every caption to this many tokens
                (notebook: ``MAX_LENGTH``).
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self._layer = None
        self._idx2word = None
        self._word2idx = None

    # ----------------------------------------------------------------- fit ----

    def fit(self, captions: Iterable[str]) -> None:
        """Adapt the underlying TextVectorization layer to the given captions.

        Args:
            captions: An iterable of *already preprocessed* captions
                (i.e. lower-cased, punctuation-stripped, wrapped in
                ``[start] ... [end]``). Mirrors notebook cell 7 which calls
                ``tokenizer.adapt(captions['caption'])`` *after* cell 4 has
                applied ``preprocess`` to every row.
        """
        import tensorflow as tf

        layer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            standardize=None,
            output_sequence_length=self.max_length,
        )
        layer.adapt(list(captions))
        self._layer = layer
        self._build_lookups()

    # ----------------------------------------------------------- properties ---

    @property
    def vocabulary(self) -> list[str]:
        """Return the fitted vocabulary list (same order as TextVectorization)."""
        layer = self._require_fit()
        return list(layer.get_vocabulary())

    @property
    def vocabulary_size(self) -> int:
        """Number of tokens in the fitted vocabulary."""
        return int(self._require_fit().vocabulary_size())

    @property
    def layer(self):
        """Direct access to the inner Keras layer.

        Exposed because the model's ``Embeddings`` layer (notebook cell 19)
        needs ``tokenizer.vocabulary_size()`` at construction time. Phase 1b
        replaces this with a constructor argument and removes the property.
        """
        return self._require_fit()

    # -------------------------------------------------------- encode/decode ---

    def encode(self, text):
        """Encode ``text`` (str or list[str]) to integer-id tensor.

        Mirrors ``tokenizer(text)`` in notebook cells 7 and 25. Single string
        returns a 1-D tensor of shape ``[max_length]``; list returns 2-D.
        """
        return self._require_fit()(text)

    def decode_id(self, idx) -> str:
        """Inverse-lookup a single integer id to its string token.

        Mirrors notebook cell 25's
        ``idx2word(pred_idx).numpy().decode('utf-8')``.
        """
        self._require_fit()
        # By invariant, _idx2word is set together with _layer in fit/load.
        assert self._idx2word is not None
        word = self._idx2word(idx)
        return word.numpy().decode("utf-8")

    def word_to_id(self, word: str) -> int:
        """Look up a single word's integer id, returning 1 (the OOV id) if absent.

        Used by beam search to seed beams with the ``[start]`` token without
        going through ``TextVectorization``'s padded-string path.
        """
        self._require_fit()
        assert self._word2idx is not None
        return int(self._word2idx(word).numpy())

    # ---------------------------------------------------------- persistence ---

    def save(self, directory: str | Path) -> None:
        """Save the vocabulary to ``directory/vocab.pkl`` and ``vocab.json``.

        The pickle matches notebook cell 9 exactly so old artefacts remain
        loadable. The JSON sidecar is human-inspectable.
        """
        self._require_fit()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        vocab = self.vocabulary
        with (directory / VOCAB_PICKLE_FILENAME).open("wb") as f:
            pickle.dump(vocab, f)
        with (directory / VOCAB_JSON_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        vocab_size: int,
        max_length: int,
    ) -> CaptionTokenizer:
        """Load a previously saved vocabulary into a new tokenizer.

        Args:
            directory: Directory containing ``vocab.pkl`` (or ``vocab.json``).
            vocab_size: Maximum vocabulary size — must match the saved vocab.
            max_length: Pad/truncate length — must match training-time value.

        Returns:
            A fitted ``CaptionTokenizer`` ready to ``encode`` and ``decode_id``.
        """
        import tensorflow as tf

        directory = Path(directory)
        pkl = directory / VOCAB_PICKLE_FILENAME
        js = directory / VOCAB_JSON_FILENAME
        if pkl.is_file():
            with pkl.open("rb") as f:
                vocab = pickle.load(f)
        elif js.is_file():
            with js.open(encoding="utf-8") as f:
                vocab = json.load(f)
        else:
            raise FileNotFoundError(
                f"No tokenizer vocabulary found in {directory!s}. "
                f"Expected '{VOCAB_PICKLE_FILENAME}' (preferred) or "
                f"'{VOCAB_JSON_FILENAME}'. Train the model with "
                "`python -m scripts.train --config configs/base.yaml` to "
                "produce the artefacts, or point BACKEND_TOKENIZER_DIR at a "
                "directory that contains them."
            )

        tok = cls(vocab_size=vocab_size, max_length=max_length)
        layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            standardize=None,
            output_sequence_length=max_length,
        )
        layer.set_vocabulary(vocab)
        tok._layer = layer
        tok._build_lookups()
        return tok

    # -------------------------------------------------------------- internal --

    def _build_lookups(self) -> None:
        """Construct ``StringLookup`` (idx → word) for inference decoding.

        Called only from ``fit()`` and ``load()``, *after* ``self._layer`` has
        been assigned, so the assertion below is a defensive no-op for mypy.
        """
        import tensorflow as tf

        assert self._layer is not None
        vocab = self._layer.get_vocabulary()
        self._word2idx = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocab)
        self._idx2word = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocab, invert=True)

    def _require_fit(self):
        """Validate that the tokenizer has been fitted; return the inner layer.

        Returning the layer (rather than only raising on the unfit state)
        gives callers a non-``None``-typed local for the rest of their body —
        which is what mypy needs to prove ``layer.get_vocabulary()`` etc.
        are valid calls. Costs one attribute lookup at runtime.
        """
        if self._layer is None:
            raise RuntimeError(
                "CaptionTokenizer not fitted. Call `.fit(captions)` or "
                "`.load(directory, ...)` first."
            )
        return self._layer
