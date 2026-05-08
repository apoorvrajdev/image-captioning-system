"""Utils — cross-cutting helpers used by every other sub-package.

Kept deliberately small. If a "util" grows past a single function, that's a
signal it belongs in its own package, not here.

    logging.py   structlog setup (JSON in prod, pretty in dev)
    seed.py      ``set_global_seed`` for reproducibility
    hashing.py   ``sha256_file`` for the paper-notebook freeze check
"""

from captioning.utils.hashing import sha256_file
from captioning.utils.logging import configure_logging, get_logger
from captioning.utils.seed import set_global_seed

__all__ = [
    "configure_logging",
    "get_logger",
    "set_global_seed",
    "sha256_file",
]
