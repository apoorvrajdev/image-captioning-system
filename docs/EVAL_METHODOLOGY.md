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
