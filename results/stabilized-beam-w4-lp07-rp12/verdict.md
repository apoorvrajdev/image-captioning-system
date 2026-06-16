# Evaluation-Methodology Gate — Verdict (v2.0.0 checkpoint)

**Date:** 2026-06-16
**Scope:** Stage 0 gate from `Full_Step-By-Step_Plan__Option_B_.txt` — decide whether the
3.5-hour Kaggle retrain (Stage 1) is required, optional, or unnecessary.
**Status:** Rule outcome = **FLAGGED FOR HUMAN REVIEW**. Recommendation below pending human sign-off.

---

## How this was produced

Two pre-registered, falsifiable tests, committed **before** any result and run in
**separate, blinded** steps so the BLEU number could not bias the qualitative read:

- **Part A — 5-ref BLEU rescore** (`scripts/rescore_nltk_bleu.py`): re-scored the committed
  v2.0.0 predictions against the full COCO 5-reference set. Artefact: `metrics_5ref.json`.
- **Part B — blinded categorization** (`scripts/categorize_predictions.py`): 30 predictions
  judged against a pre-registered rubric without sight of Part A. Artefacts: `categories.jsonl`,
  `qualitative_categorized.jsonl`.

---

## 1. Part A band and supporting values

| Quantity | Value |
|---|---|
| **Band** | **DOMINANT** (5-ref sacrebleu corpus BLEU-4 ≥ 18) |
| 5-ref sacrebleu corpus **BLEU-4** | **25.91** |
| 5-ref NLTK method1 BLEU-4 | 22.22 |
| 5-ref NLTK method4 BLEU-4 | 23.80 |
| Prior committed **1.46-ref** BLEU-4 | 10.39 |
| IEEE paper baseline | ~24 |
| Reference join | mean 5.0 refs/image, min 5, max 5, 0 images with <5 refs (n=500) |
| Secondary check | method1 − corpus BLEU-4 = **−3.68** → `axis_a_wash_under_5ref = false` |

**Finding:** the reference-count axis alone lifted BLEU-4 from **10.39 → 25.91 (+15.5)**, at/above the
IEEE ~24 baseline. The apparent "gap to IEEE" was almost entirely a **single-reference scoring
artefact**, not a model deficit. The pre-registered hypothesis ("reference-count is the dominant
remaining axis") is **confirmed**.

**Secondary flag (pre-registered):** at 5 refs, NLTK method1 trails sacrebleu corpus by 3.68 points
(> 3), so Axis A (aggregation/smoothing) is **not** a perfect wash under 5 refs — it contributes
~3.7 points. This does not change the band (method1 = 22.2 is still ≥ 18) but is recorded for honesty.

## 2. Part B counts (blinded)

- **SPECIFIC-CORRECT = 3/30**
- GENERIC-CORRECT = 11/30
- PARTIALLY-CORRECT = 15/30
- INCORRECT = 1/30

(N=30 → roughly ±18% sampling margin on any proportion; counts reported, not point-estimate percentages.)

## 3. Decision-rule branch that fired

```
Band = DOMINANT (≥18):
  "DOMINANT AND SPECIFIC-CORRECT >= 12/30"  →  3 < 12  →  does NOT fire.
MAJOR-BUT-PARTIAL / MINOR branches            →  not our band  →  do NOT fire.
Catch-all:
  "Any combination not covered above (e.g. DOMINANT with SPECIFIC-CORRECT <12)
   -> Flag for human review. Do not auto-decide."  →  FIRES (rule's own worked example).
```

## 4. Why the rule flagged it — the two signals conflict

- **BLEU (Part A):** on equal-footing 5-reference eval, the checkpoint scores 25.9 BLEU-4 — it
  **matches/exceeds IEEE**. It does not underperform the paper.
- **Qualitative (Part B):** only 3/30 captions are genuinely image-specific; 15/30 carry a concrete
  error (count/color/action), 11/30 are generic, 1 incorrect. Fluent, but mostly non-specific.

Metric parity ≠ specific captions. The rule routes this to a human on purpose.

---

## Recommended verdict: **REFRAME — DO NOT RETRAIN (Stage 1)**

**Rationale**

1. **The retrain premise is falsified.** Stage 1 existed to "close the gap to IEEE ~24." Part A shows
   there is no gap once the model is evaluated the way the paper was (5 references): 25.9 ≥ ~24.
2. **Stage 1 would not fix the real weakness.** The Part B weakness is low *specificity*, which is an
   architectural limit of the frozen InceptionV3 encoder. Stage 1 retrains the **same architecture**
   (just the original recipe); it cannot meaningfully add image-specificity. That is **Phase 3**
   (modern vision backbones) territory.
3. **The honest story is stronger than a retrain.** "I found the apparent BLEU gap was an
   evaluation-methodology artefact (single- vs 5-reference), and proved it with a pre-registered,
   blinded test" is a more credible portfolio narrative than re-training to chase a non-existent gap.

**Required regardless of the path chosen**

- Correct the README "Model quality" / limitations framing: the current "BLEU-4 10.57 vs IEEE ~24"
  compares 1.46-reference scoring against the paper's 5-reference scoring (apples-to-oranges). Report
  the on-equal-footing **25.9** alongside an explicit single-reference caveat, and keep the honest
  note that caption *specificity* remains limited (a Phase 3 concern).

## Options considered

| Option | Decision | Cost | Fixes specificity? |
|---|---|---|---|
| **A. Reframe, don't retrain** (recommended) | Stage 7 reframe + README correction; specificity → Phase 3 | none | n/a (premise falsified) |
| B. Run Stage 1 as a deliberate ablation | Train base.yaml as recipe comparison, not a fix | ~3.5h GPU | no (same architecture) |
| C. Reframe now, ablation later | A now; keep B optional | none now | specificity → Phase 3 |
| D. Minimal README BLEU-framing fix only | Correct only the apples-to-oranges claim | none | no |

## Recommended next steps

1. Adopt Option A (or C). Rewrite the README "Model quality" section per the reframe.
2. Commit the Part B finalized artefact (`qualitative_categorized.jsonl`) and this `verdict.md`,
   each as its own atomic commit per the commit-granularity policy.
3. Defer any caption-specificity improvement to Phase 3 (foundation-model baselines).

> The combined decision rule did **not** auto-decide this case. This verdict records the recommended
> resolution of the human-review flag; the final call rests with the maintainer.
