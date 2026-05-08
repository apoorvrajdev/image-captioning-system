"""Caption text preprocessing.

Mirrors the IEEE notebook cell 3::

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^\\w\\s]", "", text)
        text = re.sub("\\s+", " ", text)
        text = text.strip()
        text = "[start] " + text + " [end]"
        return text

Why pull this out of the notebook:
    * It's a *pure function*: same input → same output, no side effects.
      Easiest possible thing to unit-test, and the lowest-risk module to verify
      parity on (one ``assert preprocess_caption("Hello, World!") == "[start] hello world [end]"``
      catches any divergence).
    * The same logic runs at training time AND at inference time. Centralising
      it eliminates the most common bug source in ML systems: train/serve skew.
"""

from __future__ import annotations

import re

START_TOKEN = "[start]"
END_TOKEN = "[end]"

# Pre-compiled for marginal speed (caption preprocessing is called ~600k+
# times during dataset prep). The compiled patterns also make intent obvious.
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_caption(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, wrap with sentinels.

    Behaviour is byte-for-byte identical to the notebook's ``preprocess()``.

    Args:
        text: Raw caption string (any case, may contain punctuation).

    Returns:
        Normalised caption with ``[start]`` and ``[end]`` sentinels, e.g.::

            >>> preprocess_caption("A man, riding   a Bike!")
            '[start] a man riding a bike [end]'

    Note:
        The notebook applies this function via ``DataFrame.apply``; we don't
        vectorise here because the regex compilation is the dominant cost and
        is already amortised over a single call.
    """
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"
