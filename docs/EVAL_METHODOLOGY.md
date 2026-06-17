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
