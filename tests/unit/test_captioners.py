"""Tests for the Phase 3 captioner interface and adapters (TASK-011).

Offline and dependency-free: no model downloads, no Hugging Face access, no
COCO images. ``torch`` and ``transformers`` are replaced by fakes in
``sys.modules`` (or blocked), so these tests behave the same whether or not the
``[hf]`` extra is installed.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

import captioning.baselines.base as base_module
from captioning.baselines import (
    HF_INSTALL_HINT,
    Captioner,
    CaptionerIdentity,
    CNNCaptioner,
    HFCaptioner,
    MissingHFDependencyError,
)
from captioning.config import AppConfig, CompareConfig, load_config
from captioning.evaluation.tokenization import strip_sentinels
from captioning.preprocessing.caption import preprocess_caption

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen by docs/EVAL_METHODOLOGY.md § 8.1 and § 8.4.
PROTOCOL_MODELS = [
    (
        "blip-base",
        "Salesforce/blip-image-captioning-base",
        "82a37760796d32b1411fe092ab5d4e227313294b",
    ),
    (
        "vit-gpt2",
        "nlpconnect/vit-gpt2-image-captioning",
        "dc68f91c06a1ba6f15268e5b9c13ae7a7c514084",
    ),
    ("git-base-coco", "microsoft/git-base-coco", "a13141da42abd4a8cbf283601a8104265f537cee"),
]
PROTOCOL_CNN = (
    "inceptionv3-transformer-stabilized",
    "apoorvrajdev/captioning-inceptionv3-transformer",
    "59d93b4babb16b0ac81eef598f3abc271a355cbf",
)
PROTOCOL_GENERATE_KWARGS = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 40,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}


# --------------------------------------------------------------------- config --


def test_compare_defaults_are_the_frozen_protocol() -> None:
    compare = AppConfig().compare

    assert (compare.cnn.model_id, compare.cnn.hub_repo, compare.cnn.revision) == PROTOCOL_CNN
    assert [(m.model_id, m.hub_repo, m.revision) for m in compare.baselines] == PROTOCOL_MODELS
    assert compare.baseline_decode.model_dump() == {
        **PROTOCOL_GENERATE_KWARGS,
        "precision": "float32",
    }


def test_base_yaml_states_the_compare_protocol_explicitly() -> None:
    base_yaml = REPO_ROOT / "configs" / "base.yaml"
    raw = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))

    assert raw["compare"] == AppConfig().compare.model_dump(mode="json")
    assert load_config(base_yaml).compare == AppConfig().compare


@pytest.mark.parametrize(
    "override",
    [
        {"unknown_key": 1},
        {"baseline_decode": {"num_beams": 1, "top_k": 50}},
        {"cnn": {"model_id": "x", "hub_repo": "a/b", "revision": "main"}},
        {"cnn": {"model_id": "x", "hub_repo": "a/b", "revision": "59d93b4"}},
        {"baselines": [{"model_id": "x", "hub_repo": "a/b", "revision": "0" * 40}] * 2},
        {"baseline_decode": {"precision": "float16"}},
    ],
)
def test_compare_config_rejects_unknown_keys_unpinned_revisions_and_duplicates(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        CompareConfig(**override)


# --------------------------------------------------------- common interface --


class _StubCaptioner(Captioner):
    def __init__(self, raw: list[str]) -> None:
        super().__init__(CaptionerIdentity("stub", None, None, {}))
        self.raw = raw
        self.calls: list[list[Path]] = []

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        self.calls.append(image_paths)
        return self.raw


def test_caption_normalises_through_the_existing_path() -> None:
    raw = ["A Dog, sitting on a COUCH.", "  two   people, holding umbrellas!! "]
    captioner = _StubCaptioner(raw)

    captions = captioner.caption(["a.jpg", Path("b.jpg")])

    assert captions == ["a dog sitting on a couch", "two people holding umbrellas"]
    assert captions == [strip_sentinels(preprocess_caption(r)) for r in raw]
    assert captioner.calls == [[Path("a.jpg"), Path("b.jpg")]]
    # No second implementation: the base class uses the existing functions.
    assert base_module.preprocess_caption is preprocess_caption
    assert base_module.strip_sentinels is strip_sentinels


def test_caption_handles_empty_batches_and_rejects_count_mismatches() -> None:
    captioner = _StubCaptioner(["only one"])

    assert captioner.caption([]) == []
    assert captioner.calls == []
    with pytest.raises(RuntimeError, match="1 captions for 2 images"):
        captioner.caption(["a.jpg", "b.jpg"])


# ---------------------------------------------------------------- CNN adapter --


class _FakePredictor:
    def __init__(self, **decode: Any) -> None:
        self.decode_strategy = decode.get("decode_strategy", "greedy")
        self.beam_width = decode.get("beam_width", 3)
        self.length_penalty = decode.get("length_penalty", 1.0)
        self.repetition_penalty = decode.get("repetition_penalty", 1.0)
        self.no_repeat_ngram_size = decode.get("no_repeat_ngram_size", 0)
        self.config = SimpleNamespace(model=SimpleNamespace(max_length=40))
        self.paths: list[str | Path] = []

    def predict_path(self, image_path: str | Path) -> str:
        self.paths.append(image_path)
        return f"a caption for {Path(image_path).stem}"


def test_cnn_captioner_wraps_the_predictor_one_image_at_a_time() -> None:
    predictor = _FakePredictor()
    captioner = CNNCaptioner(predictor, AppConfig().compare.cnn)

    captions = captioner.caption(["x/one.jpg", "x/two.jpg"])

    assert predictor.paths == [Path("x/one.jpg"), Path("x/two.jpg")]
    assert captions == ["a caption for one", "a caption for two"]
    identity = captioner.identity
    assert (identity.model_id, identity.hub_repo, identity.revision) == PROTOCOL_CNN
    assert dict(identity.decode_settings) == {
        "decode_strategy": "greedy",
        "beam_width": None,
        "length_penalty": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "max_length": 40,
    }


def test_cnn_captioner_records_beam_settings_as_used() -> None:
    predictor = _FakePredictor(
        decode_strategy="beam", beam_width=4, length_penalty=0.7, repetition_penalty=1.2
    )

    settings = CNNCaptioner(predictor, AppConfig().compare.cnn).identity.decode_settings

    assert settings["decode_strategy"] == "beam"
    assert (settings["beam_width"], settings["length_penalty"]) == (4, 0.7)
    assert settings["repetition_penalty"] == 1.2


def test_cnn_from_artifacts_delegates_to_caption_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    predictor = _FakePredictor()

    class FakeCaptionPredictor:
        @staticmethod
        def from_artifacts(*args: Any, **kwargs: Any) -> _FakePredictor:
            calls.append((args, kwargs))
            return predictor

    # Stand-ins so the lazy import never reaches TensorFlow.
    fake_predictor_module = types.ModuleType("captioning.inference.predictor")
    fake_predictor_module.CaptionPredictor = FakeCaptionPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "captioning.inference", types.ModuleType("captioning.inference")
    )
    monkeypatch.setitem(sys.modules, "captioning.inference.predictor", fake_predictor_module)

    config = AppConfig()
    captioner = CNNCaptioner.from_artifacts(
        "w/model.h5", "w", config, decode_strategy="beam", beam_width=4
    )

    assert calls == [
        (
            ("w/model.h5", "w", config),
            {
                "decode_strategy": "beam",
                "beam_width": 4,
                "length_penalty": None,
                "repetition_penalty": None,
                "no_repeat_ngram_size": None,
            },
        )
    ]
    assert captioner.identity.model_id == config.compare.cnn.model_id


# ----------------------------------------------------------------- HF adapter --


@pytest.mark.parametrize("index", range(len(PROTOCOL_MODELS)))
def test_hf_captioner_identity_and_generate_settings(index: int) -> None:
    compare = AppConfig().compare
    captioner = HFCaptioner(compare.baselines[index], compare.baseline_decode)

    identity = captioner.identity
    assert (identity.model_id, identity.hub_repo, identity.revision) == PROTOCOL_MODELS[index]
    assert captioner.generate_kwargs == PROTOCOL_GENERATE_KWARGS
    assert dict(identity.decode_settings) == {**PROTOCOL_GENERATE_KWARGS, "precision": "float32"}


class _FakeHF:
    """Records every call the adapter makes into ``torch`` / ``transformers``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.float32 = object()
        fake = self

        class Pixels:
            def __init__(self, batch_size: int) -> None:
                self.batch_size = batch_size

            def to(self, *args: Any, **kwargs: Any) -> Pixels:
                fake.calls.append(("pixels.to", args, kwargs))
                return self

        class ImageProcessor:
            def __call__(self, *, images: list[Image.Image], return_tensors: str) -> Any:
                fake.calls.append(
                    (
                        "image_processor",
                        (),
                        {"modes": [i.mode for i in images], "rt": return_tensors},
                    )
                )
                return SimpleNamespace(pixel_values=Pixels(len(images)))

        class Tokenizer:
            def batch_decode(self, ids: list[int], skip_special_tokens: bool) -> list[str]:
                fake.calls.append(("batch_decode", (ids,), {"skip": skip_special_tokens}))
                return ["A Dog, sitting.", "TWO cats!"][: len(ids)]

        class Model:
            def to(self, device: str) -> Model:
                fake.calls.append(("model.to", (device,), {}))
                return self

            def eval(self) -> Model:
                fake.calls.append(("model.eval", (), {}))
                return self

            def generate(self, **kwargs: Any) -> list[int]:
                fake.calls.append(("generate", (), kwargs))
                return list(range(kwargs["pixel_values"].batch_size))  # one sequence per image

        def auto(name: str, product: Any) -> Any:
            def from_pretrained(repo: str, **kwargs: Any) -> Any:
                fake.calls.append((f"{name}.from_pretrained", (repo,), kwargs))
                return product

            return SimpleNamespace(from_pretrained=from_pretrained)

        self.torch = types.ModuleType("torch")
        self.torch.float32 = self.float32  # type: ignore[attr-defined]
        self.torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
        self.transformers = types.ModuleType("transformers")
        self.transformers.AutoImageProcessor = auto("AutoImageProcessor", ImageProcessor())  # type: ignore[attr-defined]
        self.transformers.AutoTokenizer = auto("AutoTokenizer", Tokenizer())  # type: ignore[attr-defined]
        self.transformers.AutoModelForVision2Seq = auto("AutoModelForVision2Seq", Model())  # type: ignore[attr-defined]

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


def _write_images(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        path = tmp_path / f"{i}.png"
        Image.new("L", (4, 4)).save(path)  # greyscale on disk: the adapter must convert to RGB
        paths.append(path)
    return paths


def test_hf_captioner_loads_the_pinned_revision_and_generates_with_protocol_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeHF()
    monkeypatch.setitem(sys.modules, "torch", fake.torch)
    monkeypatch.setitem(sys.modules, "transformers", fake.transformers)
    compare = AppConfig().compare
    model = compare.baselines[0]
    captioner = HFCaptioner(model, compare.baseline_decode, device="cpu")

    captions = captioner.caption(_write_images(tmp_path, 2))
    captioner.caption(_write_images(tmp_path, 1))

    assert captions == ["a dog sitting", "two cats"]
    loads = [c for c in fake.calls if c[0].endswith("from_pretrained")]
    assert loads == [
        ("AutoImageProcessor.from_pretrained", (model.hub_repo,), {"revision": model.revision}),
        ("AutoTokenizer.from_pretrained", (model.hub_repo,), {"revision": model.revision}),
        (
            "AutoModelForVision2Seq.from_pretrained",
            (model.hub_repo,),
            {"revision": model.revision, "torch_dtype": fake.float32},
        ),
    ]
    assert fake.names()[3:5] == ["model.to", "model.eval"]
    assert ("model.to", ("cpu",), {}) in fake.calls
    assert ("image_processor", (), {"modes": ["RGB", "RGB"], "rt": "pt"}) in fake.calls
    assert ("pixels.to", ("cpu",), {"dtype": fake.float32}) in fake.calls
    generate = [kwargs for name, _, kwargs in fake.calls if name == "generate"]
    assert len(generate) == 2
    assert {k: v for k, v in generate[0].items() if k != "pixel_values"} == PROTOCOL_GENERATE_KWARGS
    assert all(kwargs for name, _, kwargs in fake.calls if name == "batch_decode")
    assert all(c[2] == {"skip": True} for c in fake.calls if c[0] == "batch_decode")


@pytest.mark.parametrize("missing", ["torch", "transformers"])
def test_hf_captioner_without_the_hf_extra_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    fake = _FakeHF()
    monkeypatch.setitem(sys.modules, "torch", fake.torch)
    monkeypatch.setitem(sys.modules, "transformers", fake.transformers)
    monkeypatch.setitem(sys.modules, missing, None)  # behaves as "not installed"
    compare = AppConfig().compare
    captioner = HFCaptioner(compare.baselines[0], compare.baseline_decode)

    with pytest.raises(MissingHFDependencyError, match=r'pip install -e "\.\[hf\]"') as excinfo:
        captioner.caption(["a.jpg"])
    assert isinstance(excinfo.value, ImportError)
    assert HF_INSTALL_HINT == 'pip install -e ".[hf]"'


# ----------------------------------------------------------- import boundary --


def _run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_importing_the_package_loads_no_model_dependencies() -> None:
    out = _run_python(
        "import sys\n"
        "import captioning, captioning.config, captioning.evaluation, captioning.baselines\n"
        "from captioning.baselines import HFCaptioner\n"
        "from captioning.config import AppConfig\n"
        "c = AppConfig().compare\n"
        "[HFCaptioner(m, c.baseline_decode) for m in c.baselines]\n"
        "print(sorted(m for m in ('torch', 'transformers', 'tensorflow') if m in sys.modules))\n"
    )
    assert out == "[]"


def test_package_imports_and_fails_clearly_when_hf_is_not_installed() -> None:
    out = _run_python(
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        "from captioning.baselines import HFCaptioner, MissingHFDependencyError\n"
        "from captioning.config import AppConfig\n"
        "c = AppConfig().compare\n"
        "captioner = HFCaptioner(c.baselines[0], c.baseline_decode)\n"
        "try:\n"
        "    captioner.load()\n"
        "except MissingHFDependencyError as exc:\n"
        "    print(exc)\n"
    )
    assert 'pip install -e ".[hf]"' in out
