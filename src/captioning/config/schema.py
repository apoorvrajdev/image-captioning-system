"""Typed configuration schemas (Pydantic v2 ``BaseSettings``).

These classes replace the bare globals ``MAX_LENGTH``, ``BATCH_SIZE``, ... that
the notebook holds in cell 6. The advantages of doing this:

1. **Type safety** — every field has a declared type and Pydantic validates
   it at load time. A YAML typo (``batch_size: "64"`` as a string) raises an
   error pointing at the file and field, not a mysterious training failure
   six steps later.
2. **Env override** — ``CAPTIONING__TRAIN__BATCH_SIZE=32`` overrides
   ``train.batch_size`` without editing YAML. The double underscore is the
   nesting delimiter (configurable below). Useful for CI smoke tests.
3. **Single source of truth** — every other module accepts a sub-config
   (``ModelConfig``, ``TrainConfig``, ...) instead of pulling globals. That
   makes them testable in isolation and trivially overridable in serve.

The schema mirrors the IEEE notebook 1:1 — same field names where reasonable,
same default values. Extending it (Phase 1b: warmup/cosine LR; Phase 3: model
registry) only adds new fields, never changes the meaning of existing ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _StrictModel(BaseModel):
    """Shared base for every sub-config — rejects unknown keys.

    Pydantic's default ``extra="ignore"`` silently drops misspelled fields.
    For configs that drive ML hyperparameters that's the worst possible
    behaviour: a typo (``vocabularsy_size`` instead of ``vocabulary_size``)
    silently uses the default and the model trains with the wrong value.
    Forbidding extras turns every typo into a load-time error pointing at
    the offending field.

    Note: ``extra="forbid"`` is set on ``AppConfig`` separately because
    ``BaseSettings`` uses ``SettingsConfigDict``, not ``ConfigDict``.
    """

    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    """Where the dataset lives and how much of it to use.

    Attributes:
        base_path: Root of the COCO dataset. Mirrors the notebook's
            ``BASE_PATH = '../input/coco-2017-dataset/coco2017'``.
        annotations_filename: Name of the captions JSON inside ``annotations/``.
        images_subdir: Sub-folder under ``base_path`` containing JPEGs.
        sample_size: How many caption pairs to sample. The notebook samples
            120k. Set to ``-1`` to use the full set.
        train_val_split: Fraction of *images* (not captions) used for training.
            Splitting at the image level prevents the same image appearing in
            both splits via different captions — a real leakage source.
    """

    base_path: Path = Path("data/coco2017")
    annotations_filename: str = "captions_train2017.json"
    images_subdir: str = "train2017"
    sample_size: int = 120_000
    train_val_split: float = 0.8

    @field_validator("train_val_split")
    @classmethod
    def _validate_split(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"train_val_split must be in (0, 1), got {v}")
        return v


class ModelConfig(_StrictModel):
    """Architecture hyperparameters.

    Defaults match the IEEE paper / notebook cell 6 exactly. Changing any of
    these requires re-training and re-publishing the model card on HF Hub.
    """

    embedding_dim: int = 512
    units: int = 512
    max_length: int = 40
    vocabulary_size: int = 15_000
    encoder_num_heads: int = 1  # Notebook cell 21: TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder_num_heads: int = 8  # Notebook cell 21: TransformerDecoderLayer(..., 8)
    decoder_dropout_inner: float = 0.3  # Notebook cell 19: dropout_1
    decoder_dropout_outer: float = 0.5  # Notebook cell 19: dropout_2
    decoder_attention_dropout: float = 0.1  # Notebook cell 19: MultiHeadAttention(dropout=0.1)


class TrainConfig(_StrictModel):
    """Optimisation hyperparameters.

    The Phase 1 baseline mirrors the IEEE notebook: constant LR, no label
    smoothing, dropout-active validation (a notebook quirk preserved for
    parity). The fields below the comment line are *opt-in* training-
    stability knobs added during the caption-quality stabilisation phase.
    Defaults keep every existing run byte-for-byte identical to the
    notebook; flipping the flag in YAML opts a run into the modern recipe.
    """

    epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 1_000  # tf.data shuffle buffer
    early_stopping_patience: int = 3
    seed: int = 42  # NEW (not in notebook): pin RNGs for reproducibility
    learning_rate: float = 1e-3  # Notebook uses Keras Adam default == 1e-3
    weights_filename: str = "model.h5"

    # ---- opt-in stability flags (default values preserve notebook parity) ----
    label_smoothing: float = 0.0
    lr_schedule: str = "constant"  # "constant" | "cosine"
    warmup_steps: int = 0
    cosine_decay_steps: int | None = None  # If None, derived from epochs * steps_per_epoch
    min_learning_rate: float = 0.0
    honour_training_flag_in_test_step: bool = False  # parity-quirk override

    @field_validator("label_smoothing")
    @classmethod
    def _validate_label_smoothing(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {v}")
        return v

    @field_validator("lr_schedule")
    @classmethod
    def _validate_lr_schedule(cls, v: str) -> str:
        if v not in {"constant", "cosine"}:
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {v!r}")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def _validate_warmup_steps(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {v}")
        return v


class ServeConfig(_StrictModel):
    """Settings for the FastAPI backend (Phase 2). Defined here so the schema
    is complete and tests don't have to mock a sub-config's existence.

    Decoding-related defaults are deliberately conservative: ``greedy`` stays
    the default for byte-for-byte parity with the IEEE notebook. Switching to
    beam at deploy time is a one-line YAML override:

        serve:
          decode_strategy: beam
          beam_width: 4
          length_penalty: 0.7
          repetition_penalty: 1.1
          no_repeat_ngram_size: 3
    """

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    decode_strategy: str = "greedy"
    beam_width: int = 3
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("decode_strategy")
    @classmethod
    def _validate_decode_strategy(cls, v: str) -> str:
        if v not in {"greedy", "beam"}:
            raise ValueError(f"decode_strategy must be 'greedy' or 'beam', got {v!r}")
        return v

    @field_validator("beam_width")
    @classmethod
    def _validate_beam_width(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"beam_width must be >= 1, got {v}")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def _validate_repetition_penalty(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0 (1.0 disables it), got {v}")
        return v


class ComparedModelConfig(_StrictModel):
    """One model in the Phase 3 comparison, pinned to an exact Hub commit.

    ``revision`` must be a full 40-character commit SHA, never a branch or tag
    name, so a moved branch can't silently change the model being evaluated.
    """

    # ``model_id`` matches the run_meta.json field; opt out of pydantic's
    # protected ``model_`` namespace warning, as the API response schemas do.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(min_length=1)
    hub_repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class BaselineDecodeConfig(_StrictModel):
    """Generation settings passed explicitly to every Hugging Face baseline.

    Greedy decoding, so each repository's own generation defaults never apply
    (``docs/EVAL_METHODOLOGY.md`` § 8.4).
    """

    num_beams: int = Field(default=1, ge=1)
    do_sample: bool = False
    max_new_tokens: int = Field(default=40, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)
    precision: Literal["float32"] = "float32"


def _protocol_cnn() -> ComparedModelConfig:
    return ComparedModelConfig(
        model_id="inceptionv3-transformer-stabilized",
        hub_repo="apoorvrajdev/captioning-inceptionv3-transformer",
        revision="59d93b4babb16b0ac81eef598f3abc271a355cbf",  # tag v2.0.0
    )


def _protocol_baselines() -> list[ComparedModelConfig]:
    return [
        ComparedModelConfig(
            model_id="blip-base",
            hub_repo="Salesforce/blip-image-captioning-base",
            revision="82a37760796d32b1411fe092ab5d4e227313294b",
        ),
        ComparedModelConfig(
            model_id="vit-gpt2",
            hub_repo="nlpconnect/vit-gpt2-image-captioning",
            revision="dc68f91c06a1ba6f15268e5b9c13ae7a7c514084",
        ),
        ComparedModelConfig(
            model_id="git-base-coco",
            hub_repo="microsoft/git-base-coco",
            revision="a13141da42abd4a8cbf283601a8104265f537cee",
        ),
    ]


class CompareConfig(_StrictModel):
    """Phase 3 model comparison (ADR-019).

    The defaults are the frozen protocol in ``docs/EVAL_METHODOLOGY.md`` § 8.1
    (models and revisions) and § 8.4 (baseline decoding). Changing them is a
    protocol amendment, not a tuning knob. The CNN + Transformer's own decoding
    still comes from ``serve`` / the caller, exactly as in ``scripts/evaluate.py``.
    """

    cnn: ComparedModelConfig = Field(default_factory=_protocol_cnn)
    baselines: list[ComparedModelConfig] = Field(default_factory=_protocol_baselines)
    baseline_decode: BaselineDecodeConfig = Field(default_factory=BaselineDecodeConfig)

    @model_validator(mode="after")
    def _validate_unique_model_ids(self) -> CompareConfig:
        ids = [self.cnn.model_id, *(m.model_id for m in self.baselines)]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"model_id must be unique across compared models, got {duplicates}")
        return self


class AppConfig(BaseSettings):
    """Top-level config aggregating every sub-config.

    Loaded by ``captioning.config.loader.load_config(yaml_path)``. Env vars
    with prefix ``CAPTIONING__`` override fields at any depth.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)

    model_config = SettingsConfigDict(
        env_prefix="CAPTIONING__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",  # Reject unknown keys — catches typos at load time
    )
