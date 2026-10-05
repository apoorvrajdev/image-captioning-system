"""Phase 3 comparison models behind one captioning interface (ADR-019).

Every compared model (the project's CNN + Transformer and the pretrained
Hugging Face baselines) captions images through :class:`Captioner`, so the
comparison runner and the latency benchmark share one call path and one
caption normalisation.

``transformers`` and ``torch`` belong to the optional ``[hf]`` extra and are
imported only inside :meth:`HFCaptioner.load`. Importing this package loads
neither of them, and doesn't load TensorFlow either.
"""

from captioning.baselines.base import Captioner, CaptionerIdentity
from captioning.baselines.cnn import CNNCaptioner
from captioning.baselines.hf import HF_INSTALL_HINT, HFCaptioner, MissingHFDependencyError

__all__ = [
    "HF_INSTALL_HINT",
    "CNNCaptioner",
    "Captioner",
    "CaptionerIdentity",
    "HFCaptioner",
    "MissingHFDependencyError",
]
