# Notebooks

This directory holds Jupyter notebooks. Each notebook has a specific role in
the project lifecycle, and the rules are different for each one.

---

## `01_ieee_inceptionv3_transformer.ipynb` — **FROZEN**

This notebook is the **canonical research artefact** behind the IEEE
publication [*AI Narratives: Bridging Visual Content and Linguistic
Expression*](https://ieeexplore.ieee.org/document/10675203). It contains the
exact training pipeline, hyperparameters, and inference code used to produce
the BLEU ~24 score reported in the paper.

### Why is it frozen?

Reproducibility of a published result is non-negotiable. If the notebook drifts
from what the paper describes, anyone trying to reproduce the result —
reviewers, future students, recruiters running the demo — will see numbers that
don't match the paper. That breaks scientific trust.

### Rules

1. **Do not edit cells.** No improvements, no refactors, no comment fixes.
2. **Do not re-run cells with different seeds.** The committed outputs are
   reference outputs — they are stripped on commit by `nbstripout`, but the
   structure must stay identical.
3. **Improvements go into the modular package** at [`src/captioning/`](../src/captioning/),
   never back into this notebook.
4. **Parity is enforced in CI.** The `make freeze-paper-notebook` target
   computes a SHA-256 of this file and asserts it matches the locked hash in
   `.paper-notebook.sha256`. If you change a cell, CI fails until you either
   revert OR explicitly re-lock with `make lock-paper-notebook` AND update
   the paper / model card to reflect the new behaviour.

### When this rule changes

The frozen state lifts when (and only when) we publish a v2 of the paper or
explicitly mark a re-run in the changelog. Until then, treat this file like
a museum exhibit.

---

## `02_dataset_eda.ipynb` — exploratory (Phase 1+)

Dataset inspection. Caption length distributions, vocabulary coverage, image
dimension histograms, class balance across COCO super-categories. This
notebook **may** be edited freely; it's a working scratchpad, not a published
artefact.

## `03_attention_visualization.ipynb` — exploratory (Phase 4+)

Visualisations of decoder attention weights over image patches. Used to
generate the figures in [`docs/results/`](../docs/results/). Outputs are
stripped by `nbstripout` on commit; PNGs land in `docs/images/attention/`
when explicitly exported.

---

## Conventions for new notebooks

If you add a new notebook:

- **Number it** (`04_*`, `05_*`) so the lifecycle order is obvious.
- **Use prose Markdown cells** between code cells — a notebook reads like a
  short paper, not a Python script.
- **Do not import from `notebooks/`** elsewhere in the codebase. Notebooks
  consume the `captioning` package; they never define library code.
- **Strip outputs before committing.** `nbstripout` does this automatically
  if you ran `make install-hooks`. Without that hook, run `nbstripout
  notebooks/your_notebook.ipynb` manually before `git add`.

---

## Why notebooks at all?

Notebooks are excellent for *exploration* — narrative, mixed media, iterative
data wrangling. They are bad for *libraries* — no testing, no type-checking,
no module reuse, hidden cell-execution-order bugs. The IEEE notebook stays
because the paper points at it; everything else lives in `src/captioning/`.
