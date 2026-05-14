"""Inference — generation algorithms and the FastAPI-friendly ``CaptionPredictor``.

The notebook generates captions through a free-floating ``generate_caption``
function that closes over global state (``caption_model``, ``tokenizer``,
``MAX_LENGTH``). We keep the same algorithm but inject those dependencies
explicitly so it works inside a long-lived process (FastAPI lifespan).

    image_loader.py   ``load_image_from_path`` — used at request time
    greedy.py         ``generate_caption_greedy`` — the notebook's argmax decode loop
    predictor.py      ``CaptionPredictor`` — singleton wrapper for the API
"""

from captioning.inference.beam import generate_caption_beam
from captioning.inference.greedy import generate_caption_greedy
from captioning.inference.image_loader import load_image_from_path
from captioning.inference.predictor import CaptionPredictor

__all__ = [
    "CaptionPredictor",
    "generate_caption_beam",
    "generate_caption_greedy",
    "load_image_from_path",
]
