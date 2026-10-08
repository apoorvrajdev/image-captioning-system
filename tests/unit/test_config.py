"""Tests for the Pydantic config schema and YAML loader."""

from __future__ import annotations

import json
import os
from operator import attrgetter
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings.sources import SettingsError

from captioning.config.loader import load_config
from captioning.config.schema import AppConfig, DataConfig, ModelConfig, TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_captioning_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no ``CAPTIONING__*`` override, whatever the shell exports."""
    for name in [n for n in os.environ if n.upper().startswith("CAPTIONING__")]:
        monkeypatch.delenv(name)


def test_defaults_match_notebook_hyperparams() -> None:
    """The defaults *are* the IEEE notebook's hyperparameters; if anyone
    changes them by accident, this test fails loudly."""
    cfg = AppConfig()
    assert cfg.model.embedding_dim == 512
    assert cfg.model.units == 512
    assert cfg.model.max_length == 40
    assert cfg.model.vocabulary_size == 15_000
    assert cfg.model.encoder_num_heads == 1
    assert cfg.model.decoder_num_heads == 8
    assert cfg.train.epochs == 10
    assert cfg.train.batch_size == 64
    assert cfg.train.buffer_size == 1_000
    assert cfg.train.early_stopping_patience == 3
    assert cfg.data.sample_size == 120_000
    assert cfg.data.train_val_split == 0.8


def test_split_validation_rejects_invalid_fractions() -> None:
    with pytest.raises(ValidationError):
        DataConfig(train_val_split=0.0)
    with pytest.raises(ValidationError):
        DataConfig(train_val_split=1.0)
    with pytest.raises(ValidationError):
        DataConfig(train_val_split=1.5)


def test_extra_keys_rejected() -> None:
    """``extra="forbid"`` catches typos at load time instead of training time."""
    with pytest.raises(ValidationError):
        AppConfig(model={"embedding_dim": 512, "tpyo": True})  # type: ignore[arg-type]


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPTIONING__TRAIN__BATCH_SIZE", "32")
    cfg = AppConfig()
    assert cfg.train.batch_size == 32


def test_load_config_yaml(tmp_path: Path) -> None:
    yaml_text = """
data:
  sample_size: 1000
model:
  embedding_dim: 256
train:
  epochs: 2
  batch_size: 8
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.sample_size == 1000
    assert cfg.model.embedding_dim == 256
    assert cfg.train.epochs == 2
    # Unspecified fields take defaults
    assert cfg.model.max_length == 40


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


# ---- Env overrides outrank the YAML ------------------------------------------

_YAML_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        """
data:
  base_path: data/coco2017
train:
  epochs: 2
  batch_size: 8
serve:
  max_upload_bytes: 2048
  decode_strategy: beam
  beam_width: 3
  cors_allowed_origins:
    - http://localhost:5173
    - http://127.0.0.1:5173
compare:
  baseline_decode:
    num_beams: 1
    max_new_tokens: 40
""",
        encoding="utf-8",
    )
    return p


def test_yaml_values_apply_without_env_override(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    assert cfg.train.batch_size == 8
    assert cfg.serve.beam_width == 3
    assert cfg.serve.cors_allowed_origins == _YAML_ORIGINS


def test_env_overrides_a_yaml_scalar(yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPTIONING__TRAIN__BATCH_SIZE", "32")
    assert load_config(yaml_path).train.batch_size == 32  # converted, not the string "32"


@pytest.mark.parametrize(
    ("env_name", "env_value", "field", "expected"),
    [
        # The Kaggle runbook's override (STABILIZED_TRAINING_RUNBOOK.md, cell 4).
        (
            "CAPTIONING__DATA__BASE_PATH",
            "/kaggle/input/coco-2017-dataset/coco2017",
            "data.base_path",
            Path("/kaggle/input/coco-2017-dataset/coco2017"),
        ),
        # Three levels deep.
        (
            "CAPTIONING__COMPARE__BASELINE_DECODE__NUM_BEAMS",
            "2",
            "compare.baseline_decode.num_beams",
            2,
        ),
    ],
)
def test_env_overrides_a_nested_yaml_value(
    yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    field: str,
    expected: object,
) -> None:
    monkeypatch.setenv(env_name, env_value)
    assert attrgetter(field)(load_config(yaml_path)) == expected


def test_env_cors_origins_replace_the_base_yaml_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production's wiring: ``configs/base.yaml`` lists localhost origins only, and the
    Space's ``CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS`` (runbook § 4) adds the SPA's."""
    base_yaml = REPO_ROOT / "configs" / "base.yaml"
    space_origins = [
        "https://image-captioning-system.vercel.app",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]
    assert space_origins[0] not in load_config(base_yaml).serve.cors_allowed_origins

    monkeypatch.setenv("CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS", json.dumps(space_origins))
    # Replaced whole, not appended: base.yaml's http://localhost:3000 is gone.
    assert load_config(base_yaml).serve.cors_allowed_origins == space_origins


def test_env_override_keeps_the_rest_of_the_yaml(
    yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONING__SERVE__BEAM_WIDTH", "5")
    cfg = load_config(yaml_path)
    assert cfg.serve.beam_width == 5
    # Siblings in the same section, and other sections, keep their YAML values.
    assert cfg.serve.decode_strategy == "beam"
    assert cfg.serve.max_upload_bytes == 2048
    assert cfg.serve.cors_allowed_origins == _YAML_ORIGINS
    assert (cfg.train.epochs, cfg.train.batch_size) == (2, 8)
    assert cfg.compare.baseline_decode.max_new_tokens == 40
    # A field neither source sets keeps its default.
    assert cfg.serve.length_penalty == 1.0


def test_each_load_reads_the_current_environment(
    yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CAPTIONING__TRAIN__BATCH_SIZE", "32")
        assert load_config(yaml_path).train.batch_size == 32
    # Nothing cached: once the variable is gone, the YAML value is back.
    assert load_config(yaml_path).train.batch_size == 8


def test_invalid_env_override_fails_instead_of_being_ignored(
    yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONING__TRAIN__BATCH_SIZE", "not-a-number")
    with pytest.raises(ValidationError, match=r"train\.batch_size"):
        load_config(yaml_path)


def test_misspelled_env_override_is_rejected(
    yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONING__SERVE__BEAM_WIDHT", "4")
    with pytest.raises(ValidationError, match=r"serve\.beam_widht"):
        load_config(yaml_path)


def test_malformed_env_list_fails_without_echoing_the_value(
    yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS", "https://unquoted.example")
    with pytest.raises(SettingsError) as excinfo:
        load_config(yaml_path)
    assert "unquoted.example" not in str(excinfo.value)


def test_train_seed_default_is_42() -> None:
    """The notebook didn't seed; we did. 42 is the project default."""
    assert TrainConfig().seed == 42


def test_modelconfig_independent_of_other_sections() -> None:
    """Sub-configs should be constructible without the parent."""
    m = ModelConfig(embedding_dim=128, vocabulary_size=500)
    assert m.embedding_dim == 128
    assert m.vocabulary_size == 500
    # Defaults preserved
    assert m.max_length == 40


# ---- Opt-in stability flags ------------------------------------------------


def test_train_stability_defaults_preserve_notebook_parity() -> None:
    t = TrainConfig()
    assert t.label_smoothing == 0.0
    assert t.lr_schedule == "constant"
    assert t.warmup_steps == 0
    assert t.honour_training_flag_in_test_step is False


def test_label_smoothing_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(label_smoothing=1.0)
    with pytest.raises(ValidationError):
        TrainConfig(label_smoothing=-0.1)


def test_lr_schedule_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(lr_schedule="square_wave")


def test_decode_strategy_validates() -> None:
    from captioning.config.schema import ServeConfig

    with pytest.raises(ValidationError):
        ServeConfig(decode_strategy="nucleus")
    s = ServeConfig(decode_strategy="beam", beam_width=4)
    assert s.beam_width == 4


def test_beam_width_and_repetition_penalty_rejected_out_of_range() -> None:
    from captioning.config.schema import ServeConfig

    with pytest.raises(ValidationError):
        ServeConfig(beam_width=0)
    with pytest.raises(ValidationError):
        ServeConfig(repetition_penalty=0.5)
