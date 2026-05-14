"""Tests for the Pydantic config schema and YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from captioning.config.loader import load_config
from captioning.config.schema import AppConfig, DataConfig, ModelConfig, TrainConfig


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
