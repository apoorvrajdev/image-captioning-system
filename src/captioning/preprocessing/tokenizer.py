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
3. **A JSON sidecar.** Pickle is fast but opaque, and loading one can run
   code. We additionally write a ``vocab.json`` file (a JSON array, one token
   per line, UTF-8) so humans and other tools can inspect the vocabulary.
   ``load`` reads and validates that file first. It reads a pickle only when
   there is no JSON file, and refuses any pickle that names a global other
   than the two NumPy uses to rebuild string scalars.

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
from typing import Any

VOCAB_PICKLE_FILENAME = "vocab.pkl"
VOCAB_JSON_FILENAME = "vocab.json"

# TextVectorization's padding and OOV tokens. ``get_vocabulary()`` returns them
# at ids 0 and 1, so every saved vocabulary starts with them.
_RESERVED_TOKENS = ("", "[UNK]")

# The only globals a vocabulary pickle may name. The notebook pickles
# ``get_vocabulary()``, whose entries can be ``numpy.str_``, and unpickling
# those needs NumPy's dtype and scalar reconstructor. Neither calls arbitrary
# code: they only build a dtype and a scalar from the pickled values.
_ALLOWED_PICKLE_GLOBALS = frozenset({("numpy", "dtype"), ("numpy.core.multiarray", "scalar")})


class _VocabularyUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global outside ``_ALLOWED_PICKLE_GLOBALS``."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return super().find_class(module, name)


def read_vocabulary(directory: str | Path, vocab_size: int) -> list[str]:
    """Read and validate a saved vocabulary list, without importing TensorFlow.

    ``vocab.json`` wins when both files exist. ``vocab.pkl`` is read only when
    there is no JSON file, through ``_VocabularyUnpickler``.

    Args:
        directory: Directory containing ``vocab.json`` or ``vocab.pkl``.
        vocab_size: Maximum vocabulary size the model was built with.

    Returns:
        The tokens in id order, as plain ``str``.

    Raises:
        FileNotFoundError: Neither file exists.
        ValueError: The file can't be parsed, or its list isn't a vocabulary
            this tokenizer could have saved.
    """
    directory = Path(directory)
    js = directory / VOCAB_JSON_FILENAME
    pkl = directory / VOCAB_PICKLE_FILENAME
    if js.is_file():
        source = js
        try:
            with js.open(encoding="utf-8") as f:
                vocab = json.load(f)
        except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
            raise ValueError(f"Could not parse tokenizer vocabulary {js}: {exc}") from exc
    elif pkl.is_file():
        source = pkl
        try:
            with pkl.open("rb") as f:
                vocab = _VocabularyUnpickler(f).load()
        except Exception as exc:  # malformed pickles raise many exception types
            raise ValueError(f"Could not read tokenizer vocabulary {pkl}: {exc}") from exc
    else:
        raise FileNotFoundError(
            f"No tokenizer vocabulary found in {directory!s}. "
            f"Expected '{VOCAB_JSON_FILENAME}' (preferred) or "
            f"'{VOCAB_PICKLE_FILENAME}'. Train the model with "
            "`python -m scripts.train --config configs/base.yaml` to "
            "produce the artefacts, or point BACKEND_TOKENIZER_DIR at a "
            "directory that contains them."
        )
    return _validate_vocabulary(vocab, source, vocab_size)


def _validate_vocabulary(vocab: object, source: Path, vocab_size: int) -> list[str]:
    """Return ``vocab`` as plain strings, or raise ``ValueError`` naming ``source``."""
    if not isinstance(vocab, list):
        raise ValueError(
            f"Tokenizer vocabulary {source} must be a list of strings, "
            f"got {type(vocab).__name__}."
        )
    for index, token in enumerate(vocab):
        if not isinstance(token, str):
            raise ValueError(
                f"Tokenizer vocabulary {source} has a non-string entry at id {index}: "
                f"{type(token).__name__}."
            )
    tokens = [str(token) for token in vocab]  # numpy.str_ -> str
    if tuple(tokens[:2]) != _RESERVED_TOKENS:
        raise ValueError(
            f"Tokenizer vocabulary {source} must start with the padding and OOV tokens "
            f"{list(_RESERVED_TOKENS)!r}, got {tokens[:2]!r}."
        )
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            raise ValueError(f"Tokenizer vocabulary {source} repeats the token {token!r}.")
        seen.add(token)
    if len(tokens) > vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {source} has {len(tokens)} entries, "
            f"more than vocab_size={vocab_size}."
        )
    return tokens


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
            directory: Directory containing ``vocab.json`` (preferred) or
                ``vocab.pkl``.
            vocab_size: Maximum vocabulary size — must match the saved vocab.
            max_length: Pad/truncate length — must match training-time value.

        Returns:
            A fitted ``CaptionTokenizer`` ready to ``encode`` and ``decode_id``.

        Raises:
            FileNotFoundError: Neither vocabulary file exists.
            ValueError: The vocabulary is malformed (see ``read_vocabulary``).
        """
        vocab = read_vocabulary(directory, vocab_size)

        import tensorflow as tf

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
