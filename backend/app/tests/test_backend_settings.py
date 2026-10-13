"""``BackendSettings`` imports cleanly and keeps its ``BACKEND_*`` field names."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import BackendSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]


def test_importing_the_settings_emits_no_user_warning() -> None:
    """A fresh interpreter, because the warning fires once, when the class is defined."""
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", "import app.core.config"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_version_still_reads_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BACKEND_MODEL_VERSION", "v9.9.9")

    assert BackendSettings().model_version == "v9.9.9"
