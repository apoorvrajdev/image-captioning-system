# Evaluation-Methodology Audit — Pre-Registered, Blinded BLEU Investigation

> **TL;DR.** The deployed checkpoint appeared to underperform the IEEE baseline
> by ~14 BLEU-4 points (10.4 vs ~24). Rather than spend GPU hours retraining, I
> ran a **pre-registered, blinded** evaluation audit. It found that **most of the
> apparent gap was an evaluation-methodology artefact — reference count — not a
> model deficit**: the *same* predictions score **25.9 BLEU-4** against COCO's
> full 5-reference set. A separate blinded qualitative review showed the model's
> real remaining weakness is **caption specificity**, which is architectural.
> Verdict: **reframe, do not retrain.**

This document is the methods write-up for the Stage 0 gate. All artefacts and
the pre-registration are committed under
[`results/stabilized-beam-w4-lp07-rp12/`](../results/stabilized-beam-w4-lp07-rp12/).

---

## 1. The question

The stabilized v2.0.0 checkpoint reports corpus **BLEU-4 = 10.39** (beam) on the
project's evaluation slice, against the IEEE paper's reported **~24**. The naive
reading is "the model is ~14 BLEU points worse than the paper, so retrain."

Before acting on that, one question had to be answered honestly: **is the gap a
model-quality deficit, or an evaluation-methodology artefact?** A raw BLEU delta
between two setups is not a model verdict until the evaluation is held constant.

## 2. Why a pre-registered, blinded audit

Retraining is expensive (a ~3.5-hour Kaggle run) and, more importantly, it would
have answered the *wrong* question if the gap were methodological. So instead of
retraining, the gap itself was put under test — with two guards against fooling
myself:

- **Pre-registration.** Both tests embed their hypotheses, decision thresholds,
  and rubric **verbatim in the script docstrings**, and those scripts were
  **committed before any result was produced**. Thresholds cannot be tuned
  post-hoc to fit the answer.
- **Blinding.** The quantitative (BLEU) and qualitative (human-judgment) tests
  run as **two independent scripts in two separate sessions**, sharing no code
  and no state. The qualitative categorisation was performed **before** the BLEU
  result was unblinded, so the number could not bias the judgments.

## 3. Part A — 5-reference BLEU rescore

**Script:** [`scripts/rescore_nltk_bleu.py`](../scripts/rescore_nltk_bleu.py)
· **Run it:** `make rescore-5ref COCO_ANNOTATIONS=<captions_train2017.json>`

**Hypothesis (pre-registered):** reference count is the dominant remaining axis
of the gap. The committed evaluation slice averages only **~1.46 references per
image** (most images have a single reference), whereas the IEEE-era baseline and
standard COCO scoring use the full **5 references per image**. BLEU credits an
n-gram that matches *any* reference, so reference count alone moves the score.

**Method:** re-score the *identical* beam predictions against the full COCO
5-reference set joined from `captions_train2017.json`. References are run through
the same normalisation as training so the only thing that changes is the
reference *count* — not tokenisation.

**Pre-registered decision bands** (on 5-ref sacrebleu corpus BLEU-4):

| Band | Threshold | Meaning |
|---|---|---|
| DOMINANT | ≥ 18 | methodology (reference count) dominates the gap |
| MAJOR-BUT-PARTIAL | 14–18 | methodology is a major but partial factor |
| MINOR | ≤ 13 | checkpoint genuinely underperforms |

**Result:**

| BLEU-4 (beam, identical predictions) | References / image | Source |
|---|---|---|
| 10.39 | ~1.46 (stored slice) | `metrics.json` |
| **25.91** | 5 (full COCO) | `metrics_5ref.json` (sacrebleu corpus) |

→ **Band = DOMINANT.** The reference-count axis alone lifts BLEU-4 from 10.4 to
25.9, i.e. into the IEEE baseline's range. This is **methodology parity with the
paper's evaluation setup — not a claim of superiority over it.** The headline
BLEU number is dominated by how many references you score against.

**Secondary (pre-registered) check:** at 5 references, NLTK smoothed
sentence-BLEU-4 (method1) = 22.2 trails the sacrebleu corpus value by ~3.7
points, so the aggregation/smoothing axis is **not** perfectly negligible under
the 5-reference condition — a measurable, second-order contributor. Reference
count remains the dominant factor; smoothing/aggregation is secondary.

## 4. Part B — blinded qualitative review

**Script:** [`scripts/categorize_predictions.py`](../scripts/categorize_predictions.py)
· **Run it:** `make categorize-30 COCO_ANNOTATIONS=<captions_train2017.json>`

BLEU parity does not imply good captions. Part B characterises caption quality
**independently of BLEU**, via a blinded categorisation of 30 predictions against
a pre-registered four-way rubric (each prediction gets exactly one label):

- **SPECIFIC-CORRECT** — correct subject *and* a distinguishing attribute
  (colour / count / named action / spatial relation / named secondary object).
- **GENERIC-CORRECT** — correct subject, no distinguishing detail.
- **PARTIALLY-CORRECT** — a real element captured, another wrong.
- **INCORRECT** — main subject misidentified, or scene absent from all references.

The 30-sample worklist (predictions + 5 references, **no metrics shown**) was
prepared by the script; the judgments were made by hand against the rubric and
written to [`categories.jsonl`](../results/stabilized-beam-w4-lp07-rp12/categories.jsonl),
then validated and aggregated into
[`qualitative_categorized.jsonl`](../results/stabilized-beam-w4-lp07-rp12/qualitative_categorized.jsonl).

**Result (N = 30, ~±18% sampling margin per proportion):**

| Category | Count |
|---|---|
| SPECIFIC-CORRECT | **3/30** |
| GENERIC-CORRECT | 11/30 |
| PARTIALLY-CORRECT | 15/30 |
| INCORRECT | 1/30 |

Read together: captions are **fluent and usually on-topic**, but **often generic
rather than image-specific**, and **count / colour / attribute mistakes remain
common**. **Specificity is the primary remaining weakness.**

## 5. Combined decision rule and verdict

The pre-registered combined rule takes the Part A band and the Part B
SPECIFIC-CORRECT count:

```
DOMINANT (≥18) AND SPECIFIC-CORRECT ≥ 12/30   → reframe, don't retrain
MAJOR-BUT-PARTIAL (14–18) AND ≥ 15/30          → ship without retraining
MAJOR-BUT-PARTIAL (14–18) AND < 10/30          → retrain
MINOR (≤13)                                    → retrain regardless
anything else (e.g. DOMINANT with < 12/30)     → flag for human review
```

With **band = DOMINANT** and **SPECIFIC-CORRECT = 3/30**, the catch-all branch
fired → **flag for human review** (the rule's own worked example). The two
signals conflict on purpose: BLEU says the checkpoint is at parity once the
evaluation is held constant, while the qualitative review says specificity is
weak.

**Recorded verdict** ([`verdict.md`](../results/stabilized-beam-w4-lp07-rp12/verdict.md)):
**REFRAME — do not run Stage 1 retraining.** Rationale:

1. The retrain premise ("underperforms the paper") is **falsified** — on
   equal-footing 5-reference evaluation the checkpoint is at ~25.9, in range.
2. The real weakness (specificity) is an **architectural** limit of the frozen
   InceptionV3 encoder; re-running the original training recipe under the same
   architecture would not fix it. That is a **Phase 3** concern (modern vision
   backbones).
3. The honest methodology finding is a stronger result than chasing a
   non-existent gap.

The originally-planned retrain (Option B / Stage 1) is **not deleted** — it is
**deferred to an optional, decoupled ablation**: *would the original recipe
outperform the stabilized recipe under this exact architecture?*

## 6. Reproducibility

Everything is committed and reproducible:

- Pre-registration blocks live in the two script docstrings and landed in
  history **before** any result (commits `feat(eval): …`).
- `make rescore-5ref COCO_ANNOTATIONS=…` reproduces Part A and writes
  `metrics_5ref.json` with the band.
- `make categorize-30 COCO_ANNOTATIONS=…` reproduces the blinded Part B worklist.
- Artefacts: `metrics.json`, `metrics_5ref.json`, `predictions.jsonl`,
  `categories.jsonl`, `qualitative_categorized.jsonl`, `verdict.md` — all under
  [`results/stabilized-beam-w4-lp07-rp12/`](../results/stabilized-beam-w4-lp07-rp12/).

## 7. Lessons

- **Audit the gap before closing it.** The cheapest, most honest move was an
  experiment, not a retrain — and it changed the conclusion entirely.
- **Reported BLEU is an evaluation artefact as much as a model property.**
  Reference count, tokenisation, and smoothing each move it by points; a raw
  BLEU delta across setups is not a model-quality verdict.
- **Metric parity ≠ caption quality.** Holding methodology constant closed the
  BLEU gap but surfaced specificity as the genuine, separable weakness — which
  the qualitative review was designed to catch.
- **Pre-registration + blinding are cheap insurance** against fitting the
  analysis to the answer you hoped for.

## 8. Phase 3 comparison protocol (pre-registered)

This section fixes how Phase 3 compares the project's CNN + Transformer with
three pretrained Hugging Face captioners (TASK-009 – TASK-016 in
[`TASKS.md`](TASKS.md)). It was committed on 2026-10-05, **before any baseline
result existed**; at that point `results/` held only the two CNN + Transformer
runs. Where the code lives and how it imports its dependencies:
[ADR-019](DECISIONS.md).

### 8.1 Models

| `model_id` | Hub repository | Pinned revision | Licence |
|---|---|---|---|
| `inceptionv3-transformer-stabilized` | `apoorvrajdev/captioning-inceptionv3-transformer` | tag `v2.0.0` = `59d93b4babb16b0ac81eef598f3abc271a355cbf` | MIT |
| `blip-base` | `Salesforce/blip-image-captioning-base` | `82a37760796d32b1411fe092ab5d4e227313294b` | BSD-3-Clause |
| `vit-gpt2` | `nlpconnect/vit-gpt2-image-captioning` | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | Apache-2.0 |
| `git-base-coco` | `microsoft/git-base-coco` | `a13141da42abd4a8cbf283601a8104265f537cee` | MIT |

- Revisions are each repository's `main` commit on 2026-10-05, read from the
  Hub model API (`sha`). The CNN + Transformer's is the commit behind its
  `v2.0.0` tag. Every load (model, processor, tokenizer) passes
  `revision=<sha>`.
- Licences are the `license:` field of each model card at the pinned revision,
  which the Hub also shows as the repository's `license:` tag.
- `model_id` is the value written to `run_meta.json`. The CNN + Transformer
  keeps the id its committed runs already use.

### 8.2 Evaluation slice and references

- The slice is [`results/stabilized-greedy/predictions.jsonl`](../results/stabilized-greedy/predictions.jsonl):
  500 images in file order, with their stored references (732 in total,
  1.46 per image, 315 images with a single reference).
- No re-sampling, filtering or reordering. Images are matched by file name,
  because the stored paths are Kaggle paths.
- `results/stabilized-beam-w4-lp07-rp12/predictions.jsonl` holds the same
  500 images in the same order (checked 2026-10-05).
- Every model is scored against these stored references, the same reference
  count as the existing runs. Five-reference scoring of the baselines is out of
  scope for Phase 3.

### 8.3 Caption normalisation and metrics

- Every model's raw caption goes through the existing path:
  `preprocess_caption` (`captioning/preprocessing/caption.py`), then
  `strip_sentinels` (`captioning/evaluation/tokenization.py`). The result is
  the `prediction` that is stored and scored. There is no second normalisation
  implementation.
- On the 1,000 committed CNN + Transformer predictions (greedy and beam) this
  path changes nothing (checked 2026-10-05), so the existing runs already meet
  this rule.
- Metrics come from the unchanged `compute_all_metrics`: sacrebleu corpus
  BLEU-1..4, METEOR, ROUGE-L and CIDEr, with the same code and settings as the
  existing runs.
- The artefact contract is unchanged, so only the normalised caption is stored.

### 8.4 Decoding settings

**CNN + Transformer: both decodings, greedy as the primary comparison.**

- **Primary:** greedy, the serving default (`serve.decode_strategy: greedy`).
  Its run is `results/stabilized-greedy/`. TASK-014 re-runs it through the
  harness to show the harness reproduces it.
- **Secondary:** beam width 4, length penalty 0.7, repetition penalty 1.2. Its
  run is `results/stabilized-beam-w4-lp07-rp12/`, the source of the README
  headline. It is shown as a labelled reference row. It isn't compared against
  the baselines, which are greedy only.

**Baselines: greedy, identical for all three.** The settings are passed
explicitly to `generate()`, so the repositories' own generation defaults don't
apply:

| Setting | Value |
|---|---|
| `num_beams` | 1 |
| `do_sample` | `False` |
| `max_new_tokens` | 40, matching the CNN + Transformer's `model.max_length: 40` |
| `repetition_penalty` | 1.0, as in the CNN greedy run |
| `no_repeat_ngram_size` | 0 |
| Text prompt | none (unconditional captioning) |
| Image preprocessing | each model's own processor, from the pinned revision |
| Precision | float32 |

The token cap is a safety bound, not a length match: the tokenisers differ, and
captions normally end at end-of-sequence well before 40 tokens. Seeds are set
with `set_global_seed(config.train.seed)`, as in `scripts/evaluate.py`.

### 8.5 Training-data overlap (methodological limitation)

- **Slice origin:** the slice images come from COCO `train2017`; every stored
  path is under `coco2017/train2017/`.
- **Held out for the CNN + Transformer:** training and evaluation build the same
  image-level split. Both call `make_image_level_splits` with `sample_size`
  120000, `train_val_split` 0.8 and seed 42 (see `scripts/train.py`,
  `scripts/evaluate.py` and `configs/train/stabilized.yaml`), and the slice is
  taken from the validation side.
- **The baselines' model cards, at the pinned revisions:**
  - **BLIP-base:** "Model card for image captioning pretrained on COCO dataset".
  - **GIT-base-coco:** "fine-tuned on COCO". Its pre-training pairs also include
    COCO.
  - **ViT-GPT2:** the card says the model was trained with the Hugging Face
    Flax image-captioning example and is the PyTorch version of
    `ydshieh/vit-gpt2-coco-en-ckpts`. Neither that card nor the upstream card
    names the training dataset; COCO appears only in the upstream checkpoint's
    name. It is treated as possibly trained on COCO.
- **Unmeasured overlap:** the per-image overlap between the slice and any
  baseline's training data is not measured. Some or most slice images may have
  been seen by the baselines during training.
- **Consequence:** baseline scores on this slice are **not** a held-out,
  like-for-like comparison with the CNN + Transformer and must not be presented
  as one. Every Phase 3 table, summary and dashboard states this limitation.
- **Why the protocol is kept:** the slice and reference protocol are retained
  on purpose. They are the setup behind every committed CNN + Transformer
  result, so changing them would break comparability with those runs. A slice
  held out for the baselines would be a separate, new protocol.

### 8.6 What each Phase 3 run records

- One new `results/<run_id>/` per (model, decoding), written by
  `write_run_artifacts`. Existing `results/*` directories are never modified.
- `run_meta.json` records:
  - `model_id` from § 8.1;
  - `weights_path` and `tokenizer_dir` as `<hub repository>@<revision>` for the
    baselines;
  - `decode_strategy` `greedy` with `repetition_penalty` 1.0;
  - `n_samples` 500;
  - `max_length` 40.

### 8.7 Scope and changes

- **Out of scope:** five-reference scoring of the baselines; latency (TASK-015
  sets that protocol before any timing run); fine-tuning; serving any baseline.
- **Changing this protocol:** after the first baseline result exists, any change
  to §§ 8.1–8.6 is a dated amendment that gives its reason. Runs made under the
  changed settings go to new run directories and aren't compared with runs made
  under these settings.

### 8.8 Results (TASK-014, 2026-10-06)

These are the first runs under §§ 8.1–8.6. Nothing in the protocol was changed.

**Runs.** Each was written by `scripts/compare_models.py`, one invocation per
model:

| `model_id` | Run directory | Revision |
|---|---|---|
| `blip-base` | [`results/phase3-blip-base-greedy/`](../results/phase3-blip-base-greedy/) | `82a37760796d32b1411fe092ab5d4e227313294b` |
| `vit-gpt2` | [`results/phase3-vit-gpt2-greedy/`](../results/phase3-vit-gpt2-greedy/) | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` |
| `git-base-coco` | [`results/phase3-git-base-coco-greedy/`](../results/phase3-git-base-coco-greedy/) | `a13141da42abd4a8cbf283601a8104265f537cee` |
| `inceptionv3-transformer-stabilized` (harness reproduction) | [`results/phase3-inceptionv3-transformer-stabilized-greedy/`](../results/phase3-inceptionv3-transformer-stabilized-greedy/) | `59d93b4babb16b0ac81eef598f3abc271a355cbf` (tag `v2.0.0`) |

**Execution.**

- **Host:** local CPU (Windows 11, Python 3.10.11) with `tensorflow-cpu`
  2.15.0, `torch` 2.3.0+cpu and `transformers` 4.41.2. METEOR ran through
  `pycocoevalcap` on Java 25.
- **Settings:** float32, batch size 1, device `cpu`, seed 42, with the § 8.4
  decode settings. Each run's `comparison_meta.json` records these values.
- **Host vs. plan:** `TASKS.md` planned a Kaggle session. §§ 8.1–8.6 don't fix
  the execution host, so running locally is not a protocol change.
- **Images:**
  - The 500 slice files were fetched by file name, through the Kaggle API, from
    the dataset the committed runs read (`awsaf49/coco-2017-dataset`,
    `coco2017/train2017/`). They were placed in `data/coco2017/train2017/`,
    which is not committed.
  - All 500 decode.
  - SHA-256 over the lines `<file name>\t<sha256 of file>\n`, taken in slice
    order: `ec40c17ad928548875817c5a54b78ba977b8843ae42b7f06febba70fa73eea0b`.
- **Baseline revisions:** before the runs, each baseline was loaded once at its
  pinned revision. In each case the loaded config's `_commit_hash` equalled the
  § 8.1 SHA, the weights were float32, and the classes were
  `BlipForConditionalGeneration`, `VisionEncoderDecoderModel` and
  `GitForCausalLM`.
- **CNN + Transformer weights:**
  - Source: Hub commit `59d93b4`, materialised under
    `outputs/hub/apoorvrajdev/captioning-inceptionv3-transformer@59d93b4…/`
    (the `weights_path` in its `run_meta.json`).
  - `model.h5` SHA-256:
    `74963a3f7cd01b16f44cd179f1e21e9eb8d60b52d460e517cdcf38c015689b07`.
  - The local `models/v1.0.0/` holds the development scaffold, not this
    checkpoint, so it wasn't used.

**CNN + Transformer reproduction (§ 8.4).**

- The harness run reproduces
  [`results/stabilized-greedy/`](../results/stabilized-greedy/) exactly:
  - all 500 predictions are identical strings, in the same order, with the same
    references;
  - all seven metrics are bit-identical.

  The committed run was made on Kaggle; the harness run was made on the local
  CPU.
- Rescoring the committed predictions with the local metric code:
  - reproduces the greedy metrics exactly;
  - reproduces the beam metrics to within 1e-14 (floating-point summation
    order).
- The two greedy runs share a model and decoding, so TASK-013 accepts only one
  of them per summary. The summary uses `results/stabilized-greedy/`, as § 8.4
  specifies. The harness run is kept as the reproduction record.

**Comparison summary.**
[`results/phase3-comparison/`](../results/phase3-comparison/) holds
`comparison.json` (exact values) and `comparison.md` (two decimals). It was
written by:

```bash
python -m scripts.compare_runs \
    results/phase3-blip-base-greedy results/phase3-vit-gpt2-greedy \
    results/phase3-git-base-coco-greedy \
    --reference-run results/stabilized-greedy \
    --reference-run results/stabilized-beam-w4-lp07-rp12 \
    --output-dir results/phase3-comparison
```

- **Slice check:** passed for all five runs: 500 images, 732 references,
  fingerprint `6b5628bf…`.
- **Determinism:** the output is byte-identical when the runs are given in a
  different order.

The table below is copied from `comparison.md`:

> Not a held-out comparison (§ 8.5). The slice comes from COCO `train2017`. The
> CNN + Transformer held these images out. BLIP-base and GIT-base-coco state COCO
> training, and ViT-GPT2 is treated as possibly COCO-trained. The baselines may
> have seen these images, so their scores here are not a held-out,
> like-for-like comparison with the CNN + Transformer and must not be presented
> as one.

| Run | Model | Decoding | Revision | Samples | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `phase3-blip-base-greedy` | blip-base | greedy | `82a3776` | 500 | 56.61 | 39.70 | 27.86 | 19.88 | 42.13 | 17.23 | 1.06 |
| `phase3-git-base-coco-greedy` | git-base-coco | greedy | `a13141d` | 500 | 51.59 | 36.55 | 26.05 | 18.83 | 47.08 | 21.92 | 1.46 |
| `stabilized-beam-w4-lp07-rp12` (reference) | inceptionv3-transformer-stabilized | beam | — | 500 | 41.93 | 25.41 | 16.01 | 10.39 | 36.84 | 15.56 | 0.83 |
| `stabilized-greedy` (reference) | inceptionv3-transformer-stabilized | greedy | — | 500 | 42.20 | 26.09 | 16.52 | 10.57 | 37.57 | 15.45 | 0.79 |
| `phase3-vit-gpt2-greedy` | vit-gpt2 | greedy | `dc68f91` | 500 | 49.12 | 33.41 | 22.91 | 15.84 | 44.51 | 19.78 | 1.26 |

- **Single-reference scoring:** every row is scored against the slice's stored
  references (1.46 per image). This is not the five-reference COCO setup, so
  these numbers can't be compared with published COCO results.
- **The beam row:** it is a labelled reference (§ 8.4). It isn't compared with
  the greedy baselines.

## 9. Phase 3 latency protocol (pre-registered)

This section fixes how Phase 3 times the four § 8.1 models (TASK-015, TASK-016
in [`TASKS.md`](TASKS.md)). It was committed on 2026-10-07, **before any
latency run existed**. Latency runs score nothing: §§ 8.1–8.6 and every
quality result are unchanged. The code is `captioning.evaluation.latency` and
`scripts/benchmark_latency.py`; where the artefact lives and why:
[ADR-020](DECISIONS.md).

### 9.1 What is timed

- **One sample:** the wall-clock time of one `Captioner.caption()` call on one
  batch of image paths. The call covers:
  - reading and decoding the image files;
  - each model's own preprocessing;
  - generation with the § 8.4 decode settings;
  - token decoding and the § 8.3 normalisation.

  This is the call path that produced the § 8.8 captions. There is no second
  inference path.
- **Clock:** `time.perf_counter`, Python's monotonic, highest-resolution clock.
  Only the `caption()` call sits between the two clock reads. A clock that goes
  backwards fails the run.
- **Load time:** captioner construction plus `load()`, timed once with the same
  clock and recorded as `load_seconds`. It is never part of a sample.
  - Both steps are timed together because the CNN + Transformer loads its
    checkpoint while it is constructed, and the Hugging Face adapters load in
    `load()`.
  - It includes importing TensorFlow, or `torch` and `transformers`, and any Hub
    download. It is the cold-start cost on that host, not a pure weight-read
    time. A run with an empty Hugging Face cache includes the download.
- **Not split into stages:** preprocessing, generation and normalisation are not
  timed separately. The TASK-011 interface doesn't expose them, and splitting
  them would mean changing the adapters.
- **GPU:** both adapters return decoded strings, and decoding them needs the
  generated tokens on the host. So a call can't return before its device work
  has finished, and no extra synchronisation is added.

### 9.2 Inputs

- The first N images of the § 8.2 slice, in slice order. The default N is 32.
- Every model gets the same N images.
- The slice is checked as in TASK-012 (500 images, 732 references) before any
  model loads. The run records the slice fingerprint and the N file names.
- Only those N images have to exist locally. References aren't used.

### 9.3 Warmup and measurement

- **Batch sizes:** 1 and 8 by default, run in ascending order. Each must divide
  N, so every call captions a full batch. With 32 images, a pass is 32 calls at
  batch size 1 and 4 calls at batch size 8.
- **Warmup:** for each batch size, W untimed passes (default 1) caption every
  batch.
  - Warmup runs again for each batch size, because input shapes change.
  - At least one warmup pass is required. The first call, which pays for lazy
    initialisation, graph tracing and a cold file cache, is never timed.
- **Measurement:** R timed passes (default 5) then time every call. With the
  defaults, that is 160 samples at batch size 1 and 20 at batch size 8.
- **Order:** deterministic. Batch sizes run in ascending order, passes in turn,
  and batches in slice order. Decoding draws no random numbers (§ 8.4). Seeds
  are set with `set_global_seed(config.train.seed)` anyway, as in § 8.4.
- **Failures:** the run ends and writes nothing if any call raises, or returns
  the wrong number of captions. Samples are never dropped, filtered or retried.
- **One model per invocation:** each run is its own process, so no other model
  is loaded while one is timed.

### 9.4 Statistics

For each batch size, over all of its measured samples, pooled across passes:

- count;
- mean;
- median;
- minimum;
- maximum.

All values are seconds per call, that is per batch, not per image. The raw
samples are stored in measurement order, so any other statistic can be
recomputed from the file. There are no other percentiles, no outlier removal,
no confidence intervals and no derived scores. Latency figures say nothing about
caption quality.

### 9.5 Devices and environment

- **`--device`** is `cpu` or `cuda`.
- **Hugging Face models** are moved to that `torch` device; `cuda` is the
  default GPU.
- **CNN + Transformer:** TensorFlow places it, so `--device` is checked against
  the GPUs TensorFlow can see.
  - `cpu` fails if TensorFlow can see a GPU. Hide it with
    `CUDA_VISIBLE_DEVICES=""`.
  - `cuda` fails if it can see none.
  - The list is recorded as `runtime.tensorflow_gpus`.
  - GPU runs install `tensorflow==2.15.0` in that environment only (ADR-019).
- **CNN batches:** `CNNCaptioner` captions one image at a time, through
  `predict_path` (TASK-011). A batch-8 sample is therefore eight single-image
  predictions in a row, and shows no batching gain by construction. Batch
  figures for the CNN + Transformer and for the Hugging Face models are not
  like-for-like.
- **Environment:**
  - `--environment` is free text: the owner's name for the host and hardware,
    recorded verbatim.
  - Recorded automatically: the Python version, the platform string, and the
    installed versions of `tensorflow`, `tensorflow-cpu`, `torch` and
    `transformers` (`null` when not installed).
  - Thread counts and other framework settings stay at their defaults and
    aren't recorded separately.
- **Comparability:** figures compare only within one environment and device.
  No claim is made across devices or hosts beyond the measured setups.

### 9.6 What each latency run records

- **Run directory:** one new
  `results/<prefix><model_id>-<decoding>-<device>/` per run. The default prefix
  is `phase3-latency-`, for example `results/phase3-latency-blip-base-greedy-cpu/`.
- **Contents:** only `latency.json`. No metrics, predictions or other quality
  files are written. Existing `results/*` directories are never modified.
- **Collisions:** an existing run directory is never overwritten. This is
  checked before the model loads and again before writing. A failed run writes
  nothing.
- **Fields of `latency.json`**, in this order:

  | Field | Content |
  |---|---|
  | `protocol` | `docs/EVAL_METHODOLOGY.md § 9` |
  | `model_id`, `backend`, `captioner` | the § 8.1 id; `cnn` or `huggingface`; the adapter class |
  | `hub_repo`, `revision` | the pinned § 8.1 repository and revision, checked against the config as in TASK-012 |
  | `decode_strategy`, `decode_settings` | the § 8.4 settings, as the adapter reports them |
  | `device`, `environment` | § 9.5 |
  | `inputs` | the slice source, fingerprint, and image and reference counts; the N file names in order |
  | `settings` | `num_images`, `batch_sizes`, `warmup_passes`, `measured_passes` |
  | `timing` | the clock, and the definitions of a sample and of the load time |
  | `load_seconds` | § 9.1 |
  | `batches` | per batch size: `batch_size`, `calls_per_pass`, `samples_seconds`, and `summary_seconds` (§ 9.4) |
  | `seed` | `config.train.seed` |
  | `runtime` | the Python version, platform, package versions, and `tensorflow_gpus` (CNN only, otherwise `null`) |

- **No timestamps:** given the same timings, the file is byte-identical. The
  timings themselves vary from run to run.

### 9.7 Running it (TASK-016)

One invocation per model and device, with the protocol defaults:

```bash
python -m scripts.benchmark_latency --config configs/base.yaml \
    --images-dir /path/to/coco2017/train2017 \
    --model blip-base --device cpu --environment "<host and hardware>"

python -m scripts.benchmark_latency --config configs/base.yaml \
    --images-dir /path/to/coco2017/train2017 \
    --model inceptionv3-transformer-stabilized \
    --cnn-weights <checkpoint>/model.h5 --cnn-tokenizer-dir <checkpoint> \
    --device cpu --environment "<host and hardware>"
```

`--num-images`, `--batch-size`, `--warmup-passes` and `--measured-passes` exist
for development. Runs that change them are not comparable with runs that use
the defaults.

### 9.8 Scope and changes

- **Out of scope:** serving latency on the Space (`PredictorService` reports its
  own); load testing; Prometheus; per-stage timing.
- **Changing this protocol:** after the first latency run exists, any change to
  §§ 9.1–9.6 is a dated amendment that gives its reason, as in § 8.7. Runs made
  under changed settings go to new run directories and aren't compared with runs
  made under these settings.
- **Status:** the CPU runs (§ 9.9) and the Kaggle GPU runs (§ 9.10) exist.

### 9.9 Results: CPU (TASK-016, 2026-10-07)

These are the first runs under §§ 9.1–9.6. Nothing in the protocol was changed.
The GPU runs are in § 9.10.

**Runs.** Each was written by `scripts/benchmark_latency.py`, one invocation per
model, with the protocol defaults and `--device cpu`:

| `model_id` | Run directory | Revision | Load (s) |
|---|---|---|---|
| `blip-base` | [`results/phase3-latency-blip-base-greedy-cpu/`](../results/phase3-latency-blip-base-greedy-cpu/) | `82a37760796d32b1411fe092ab5d4e227313294b` | 5.8 |
| `vit-gpt2` | [`results/phase3-latency-vit-gpt2-greedy-cpu/`](../results/phase3-latency-vit-gpt2-greedy-cpu/) | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | 8.4 |
| `git-base-coco` | [`results/phase3-latency-git-base-coco-greedy-cpu/`](../results/phase3-latency-git-base-coco-greedy-cpu/) | `a13141da42abd4a8cbf283601a8104265f537cee` | 4.6 |
| `inceptionv3-transformer-stabilized` | [`results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cpu/`](../results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cpu/) | `59d93b4babb16b0ac81eef598f3abc271a355cbf` (tag `v2.0.0`) | 5.5 |

**Execution.**

- **Host:** the owner's laptop, recorded in every run's `environment` field as:
  "Local laptop ASUS TUF Gaming A15 FA506NFR, AMD Ryzen 7 7435HS (8 cores / 16
  threads), 15.8 GiB RAM, Windows 11 Home Single Language 10.0.26200, AC power,
  power plan Turbo; CPU only (torch 2.3.0+cpu, tensorflow-cpu 2.15.0); Hugging
  Face weights from the local cache (HF_HUB_OFFLINE=1)".
  - Python 3.10.11, `transformers` 4.41.2.
  - Python 3.10 reports this host's platform as `Windows-10-10.0.26200-SP0`.
    Build 26200 is Windows 11.
  - The laptop has an NVIDIA GeForce RTX 2050, which these runs didn't use:
    `torch` is the CPU build and `tensorflow-cpu` can see no GPU
    (`runtime.tensorflow_gpus` is `[]` for the CNN).
  - The machine wasn't isolated. Background CPU load was about 1–4% before the
    runs. While they ran, no other benchmark or test was started; there were
    only brief progress checks (a process listing and log tails). Other desktop
    activity wasn't controlled.
- **Weights:**
  - The baselines loaded from the local Hugging Face cache, from the snapshots
    named by their § 8.1 revisions. `HF_HUB_OFFLINE=1` was set, so nothing was
    downloaded and the load times include no download.
  - The CNN + Transformer used the § 8.8 checkpoint (`model.h5` SHA-256
    `74963a3f…`, checked before the run).
- **Inputs:** the first 32 slice images, `000000530117.jpg` … `000000096793.jpg`
  (the full list is in each `latency.json`), from `data/coco2017/train2017/`.
  The slice fingerprint is `6b5628bf…`.
- **Run:** the four runs ran one after another. All completed; no call failed
  and nothing was retried.

**Checks on the committed files.** Each `latency.json` was checked to have:

- the § 8.1 model id, Hub repository and revision, and the § 8.4 decode
  settings;
- `device` `cpu`, the same environment, platform and package versions;
- the same 32 file names, slice fingerprint, settings and timing definition;
- batch sizes 1 and 8, with 32 and 4 calls per pass;
- 160 and 20 positive samples;
- summary statistics equal to those recomputed from the raw samples;
- no timestamp.

**Statistics.** Values come from each file's `summary_seconds`, shown here in
milliseconds per call (that is, per batch), rounded to 0.1 ms. Rows are sorted by
model id. The order is not a ranking.

| Model | Batch size | Samples | Mean | Median | Min | Max |
|---|---|---|---|---|---|---|
| blip-base | 1 | 160 | 1462.3 | 1450.0 | 862.3 | 2765.4 |
| blip-base | 8 | 20 | 11348.3 | 11043.1 | 9660.4 | 15848.8 |
| git-base-coco | 1 | 160 | 3473.5 | 3485.8 | 1736.6 | 5159.0 |
| git-base-coco | 8 | 20 | 30873.7 | 29139.3 | 24653.1 | 40414.6 |
| inceptionv3-transformer-stabilized | 1 | 160 | 1070.0 | 1052.5 | 682.8 | 1484.1 |
| inceptionv3-transformer-stabilized | 8 | 20 | 7888.8 | 7872.5 | 7583.6 | 8564.6 |
| vit-gpt2 | 1 | 160 | 851.9 | 842.3 | 704.1 | 1076.3 |
| vit-gpt2 | 8 | 20 | 4376.8 | 4331.8 | 4130.0 | 4846.3 |

- **The CNN + Transformer at batch size 8** is eight single-image predictions in
  a row (§ 9.5). It is not batched inference, and it isn't like-for-like with the
  Hugging Face models' batch-8 rows.
- **Scope of these figures:** they hold for this host, on the CPU, with these
  settings, only. They say nothing about GPU latency, other hosts or caption
  quality.
- **What a sample includes:** reading the image file and generating until each
  caption ends (§ 9.1). The spread within a row therefore includes differences
  in caption length between images.

### 9.10 Results: GPU (TASK-016, 2026-10-07)

These runs use the same protocol as § 9.9. Nothing in §§ 9.1–9.6 was changed.

**Runs.** Each was written by `scripts/benchmark_latency.py`, one invocation per
model, with the protocol defaults and `--device cuda`:

| `model_id` | Run directory | Revision | Load (s) |
|---|---|---|---|
| `blip-base` | [`results/phase3-latency-blip-base-greedy-cuda/`](../results/phase3-latency-blip-base-greedy-cuda/) | `82a37760796d32b1411fe092ab5d4e227313294b` | 3.8 |
| `vit-gpt2` | [`results/phase3-latency-vit-gpt2-greedy-cuda/`](../results/phase3-latency-vit-gpt2-greedy-cuda/) | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | 8.9 |
| `git-base-coco` | [`results/phase3-latency-git-base-coco-greedy-cuda/`](../results/phase3-latency-git-base-coco-greedy-cuda/) | `a13141da42abd4a8cbf283601a8104265f537cee` | 3.5 |
| `inceptionv3-transformer-stabilized` | [`results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cuda/`](../results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cuda/) | `59d93b4babb16b0ac81eef598f3abc271a355cbf` (tag `v2.0.0`) | 7.7 |

**Execution.**

- **Host:** the private Kaggle kernel
  `apoorvujjwal/task-016-phase-3-gpu-latency-benchmark` (version 1), with the
  `NvidiaTeslaT4` accelerator.
  - GPUs: two Tesla T4s (15360 MiB, driver 580.178.04). Every run used GPU 0
    only, through `CUDA_VISIBLE_DEVICES=0`.
  - CPU and OS: Intel Xeon @ 2.00GHz, 4 vCPUs, 31.3 GiB RAM, Linux 6.18
    (glibc 2.39).
  - Every run's `environment` field records this verbatim, together with the
    runtime below.
- **Runtime:**
  - The Kaggle image runs Python 3.13.15. `tensorflow` 2.15.0 has wheels only
    for Python 3.9–3.11, so the kernel used uv 0.11.15 to build two Python
    3.10.20 environments. Both hold the `requirements.txt` pins without
    `tensorflow-cpu`, and the repository at `03a8f9e`, installed without
    dependencies.
  - Hugging Face models: the `[hf]` extra, with `torch` 2.3.0+cu121 (CUDA 12.1)
    and `transformers` 4.41.2.
  - CNN + Transformer: `tensorflow==2.15.0` plus the 12 CUDA library pins of
    its own `and-cuda` extra. TensorFlow was built for CUDA 12.2 and cuDNN 8.
    - The extra's three TensorRT packages were left out. They can't be
      installed from PyPI, and only TF-TRT uses them; the CNN doesn't.
  - Two environments were needed because `torch` 2.3.0 and `tensorflow` 2.15.0
    pin different builds of the same CUDA libraries. For example, they require
    cuDNN 8.9.2.26 and 8.9.4.25 respectively.
  - Each run's `runtime.packages` therefore lists only its own environment's
    frameworks.
  - The repository pin `tensorflow-cpu==2.15.0` is unchanged (ADR-019).
- **GPU checks before any run:**
  - `torch` reported CUDA available and ran a convolution on `cuda:0`.
  - TensorFlow listed `/physical_device:GPU:0` (Tesla T4, compute capability
    7.5) and ran a convolution and a matrix multiply on it.
  - The CNN run records `runtime.tensorflow_gpus` as
    `["/physical_device:GPU:0"]`, so its § 9.5 device check passed.
- **Weights:**
  - Before the runs, each baseline was loaded once at its pinned revision. In
    each case the loaded config's `_commit_hash` equalled the § 8.1 SHA. The
    classes were `BlipForConditionalGeneration`, `VisionEncoderDecoderModel`
    and `GitForCausalLM`.
  - The runs then loaded from that cache, with `HF_HUB_OFFLINE=1`.
  - The CNN files came from Hub commit `59d93b4`. The SHA-256 of `model.h5`
    (`74963a3f…`) and of `vocab.pkl` (`178029c9…`) matched the Hub's LFS hashes
    before the run.
- **CNN load time:** building the CNN + Transformer first creates InceptionV3
  with Keras's ImageNet weights, which the checkpoint then overwrites.
  - The fresh Kaggle machine had no Keras cache, so this run's `load_seconds`
    includes downloading `inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5`
    (88 MB). The local CPU run (§ 9.9) had that file cached.
  - This affects the load time only, not the samples. § 9.1 counts downloads in
    the load time.
- **Inputs:** the same 32 images, read from
  `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017/`. The
  slice fingerprint is `6b5628bf…`.
- **Run:**
  - All four runs ran in the same session, one after another. All completed; no
    call failed and nothing was retried.
  - The kernel only set up the environment and called the CLI. It timed
    nothing itself.
  - The four `latency.json` files were downloaded with `kaggle kernels output`
    and committed unchanged; the SHA-256 was checked after copying.

**Checks on the committed files.** Each `latency.json` was checked to have:

- the § 8.1 model id, Hub repository and revision, and the § 8.4 decode
  settings;
- `device` `cuda`, and the same environment, platform and Python version;
- the same 32 file names, slice fingerprint, settings and timing definition;
- batch sizes 1 and 8, with 32 and 4 calls per pass;
- 160 and 20 positive samples;
- summary statistics equal to those recomputed from the raw samples;
- no timestamp.

Their inputs, settings, timing definition, revisions and decode settings also
equal those of the matching CPU run in § 9.9.

**Statistics.** Values come from each file's `summary_seconds`, shown here in
milliseconds per call (that is, per batch), rounded to 0.1 ms. Rows are sorted by
model id. The order is not a ranking.

| Model | Batch size | Samples | Mean | Median | Min | Max |
|---|---|---|---|---|---|---|
| blip-base | 1 | 160 | 178.2 | 184.6 | 97.8 | 298.9 |
| blip-base | 8 | 20 | 1102.2 | 1099.2 | 975.3 | 1235.1 |
| git-base-coco | 1 | 160 | 392.8 | 391.8 | 194.9 | 585.1 |
| git-base-coco | 8 | 20 | 3089.2 | 3103.8 | 2627.9 | 3555.5 |
| inceptionv3-transformer-stabilized | 1 | 160 | 739.4 | 724.2 | 582.1 | 1827.8 |
| inceptionv3-transformer-stabilized | 8 | 20 | 5776.5 | 5772.7 | 5556.8 | 6084.1 |
| vit-gpt2 | 1 | 160 | 198.8 | 197.7 | 158.1 | 249.9 |
| vit-gpt2 | 8 | 20 | 430.8 | 427.7 | 406.6 | 459.5 |

- **The CNN + Transformer at batch size 8** is eight single-image predictions in
  a row (§ 9.5). It is not batched inference, and it isn't like-for-like with the
  Hugging Face models' batch-8 rows.
- **Scope of these figures:** they hold for this Kaggle T4 session, with these
  settings, only.
- **CPU vs GPU:** § 9.9 and this section come from different hosts, operating
  systems and framework builds. Together they are not a controlled CPU-versus-GPU
  comparison, and no such claim is made.
