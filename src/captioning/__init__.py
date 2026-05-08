"""Captioning — production-grade extraction of the IEEE image-captioning research.

The package mirrors the IEEE notebook
(``notebooks/01_ieee_inceptionv3_transformer.ipynb``) but separates orthogonal
concerns into sub-packages so each piece is independently testable, composable,
and reusable from FastAPI / scripts.

Sub-package map:
    config/         Pydantic settings + YAML loader (the project's "type system")
    preprocessing/  Pure transforms on captions and images (no I/O, no state)
    data/           COCO loaders, splits, tf.data pipelines (I/O + statefulness)
    models/         Keras layers and models (CNN encoder + Transformer decoder)
    training/       Losses, callbacks, training orchestration
    inference/      Generation algorithms + a singleton-friendly Predictor
    evaluation/     BLEU/CIDEr/METEOR/ROUGE (Phase 1b expands these)
    utils/          Cross-cutting helpers (logging, seed, hashing, paths)

Public API is intentionally small. Everything else is internal and may change.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
