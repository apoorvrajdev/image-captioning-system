# Tasks

Executable backlog. The phase roadmap and history live in the README ([Roadmap](../README.md#-roadmap));
this file breaks the *next* work into tasks small enough to ship and review one at a time.

**Task lifecycle:** acceptance criteria → implementation → tests → verification (see
[`TEST_PLAN.md`](TEST_PLAN.md)) → review → commit → status updated here + [`MEMORY.md`](MEMORY.md).

**Format:**

```
### TASK-NNN — <imperative title>            [status: todo | in-progress | done | blocked]
Area: ml-core | inference-api | frontend | evaluation | deployment | docs
Goal: one sentence.
Acceptance criteria:
- GIVEN … WHEN … THEN …
Verification: exact commands.
```

Rule: no task spans a whole phase. If a task needs more than ~one reviewable change set, split it.

---

## Maintenance (ready)

### TASK-001 — Record the engineering-workflow bootstrap            [status: done]
Area: docs · deployment
Goal: living docs, CI parity-audit step, LF pin for the frozen notebook, CI.md drift fixes.

### TASK-002 — Decide whether to track the agent context directory            [status: done]
Area: deployment
Goal: `.claude/` is gitignored, so the repo map, skills, and lane definitions exist only on the
development machine. Decide: keep local-only, or narrow the ignore to `settings.local.json` and generated index files.
Acceptance criteria: decision recorded in `DECISIONS.md`; `.gitignore` matches it.
Outcome: tracked, with guardrails (ADR-015). Local settings and the generated index stay ignored.

### TASK-003 — Fix README drift against config and CI            [status: done]
Area: docs
Goal: README reflects `pyproject.toml` (mypy not strict), `ci.yml` (3.10/3.11 matrix + parity audit), and the
shipped checkpoint (remove "bootstrap weights" / "pending re-training" wording).
Acceptance criteria: every tool claim in README § Testing and Tech Stack matches a config file.

### TASK-004 — Reconcile model-version labelling            [status: done] (verified in production 2026-10-03)
Area: deployment · docs
Goal: one consistent version for the served checkpoint across README, `BackendSettings.model_version`
default, `.env.example`, and the Space's `BACKEND_WEIGHTS_HUB_REVISION` / `BACKEND_MODEL_VERSION`.
Acceptance criteria: `/healthz.model_version` equals the Hub tag actually served; README states the same tag.
Root cause: `BACKEND_MODEL_VERSION` was not set on the Space, so the code default `"v1.0.0"` was reported while
`BACKEND_WEIGHTS_HUB_REVISION=v2.0.0` was being served. The two settings are independent in `BackendSettings`, and
the runbook's promotion steps bumped only the revision.
Outcome:
- The owner set `BACKEND_MODEL_VERSION=v2.0.0`. Space variables (owner-confirmed):
  `BACKEND_WEIGHTS_HUB_REPO=apoorvrajdev/captioning-inceptionv3-transformer`, `BACKEND_WEIGHTS_HUB_REVISION=v2.0.0`,
  `BACKEND_MODEL_VERSION=v2.0.0`, `BACKEND_WARMUP=true`.
- Tag `v2.0.0` resolves to Hub commit `59d93b4` (trained `model.h5`, sha256 `74963a3f…`, 14,927-token vocab).
- Live `/healthz` (2026-10-03T18:12Z): HTTP 200, `model_loaded: true`, `model_version: "v2.0.0"`. The Space was
  `RUNNING` on deploy commit `123c5aa`, with no redeploy.
- README states tag `v2.0.0`. Runbook §§ 3, 4 and 10 now require both variables to move together (ADR-018).
- Not changed (out of scope, no code edits): the `BackendSettings.model_version` code default (`"v1.0.0"`, used
  only when the variable is unset) and `.env.example`.

### TASK-005 — Repair stale Makefile targets            [status: done] (verified locally 2026-10-04)
Area: deployment
Goal: `docker-build*` use the root `Dockerfile`; remove or fix `docker-up/down` (no compose file) and `eval` (missing required `--weights`/`--tokenizer-dir`).
Verification: `make -n docker-build eval` shows valid commands (dry run). On Windows: `mingw32-make -n …` (MSYS2 GNU Make).
Outcome:
- `docker-build` now runs `docker build -t captioning-backend:latest .`, which builds the root `Dockerfile` (the
  image the HF Space builds).
- `docker-build-hf` removed:
  - The Dockerfile declares no `ARG INSTALL_HF`, so the build-arg was ignored and the plain image was tagged
    `hf-latest`.
  - Nothing in `backend/`, `src/` or `scripts/` imports `transformers` or `torch`.
  - An HF image belongs to the Phase 3 task that needs one (ADR-013).
- `docker-up` / `docker-down` removed: no compose file has ever existed. The compose stack only appeared in
  `restructure-plan.md`.
- `eval` and `predict` now pass `--config configs/base.yaml`, `--weights $(MODEL_DIR)/model.h5` and
  `--tokenizer-dir $(MODEL_DIR)`, with `MODEL_DIR ?= models/v1.0.0`. That is the layout used in the README and in
  both committed `run_meta.json` files.
- `eval` dropped `--report docs/results/latest.md`, because each run already writes `results/<run_id>/report.md`.
- `predict` had the same missing-arguments defect and was included at the owner's request.
- New test `tests/unit/test_makefile.py` is static: it never runs Make or Docker and never imports TensorFlow. It
  checks Dockerfile paths, `--build-arg`/`ARG` pairs, compose targets, and that each `-m scripts.X` target passes
  the script's required options. It failed 5 checks on the old Makefile and passes 8/8 now.
- Verification:
  - `mingw32-make -n docker-build eval predict` printed the three full commands.
  - The removed targets fail with "No rule to make target" (exit 2).
  - `mingw32-make freeze-paper-notebook` printed OK.
  - click parsed the `eval` / `predict` arguments without running either script.
  - Full suite 104 passed; ruff lint + format clean; mypy 0 issues (71 files); pre-commit passed.
- Not run: `docker build` (Docker isn't installed) and a real `make eval` (no COCO data locally).
- Committed as `1be8c1f`, `29e6db5`, `a6780f6` (Makefile) and `b7b2903` (test).
- Note: locally, `models/v1.0.0` holds the dev scaffold. To evaluate the served checkpoint, pass
  `MODEL_DIR=<v2.0.0 snapshot dir>`.

### TASK-006 — Bound upload reads in `/v1/captions`            [status: done] (verified locally 2026-10-04)
Area: inference-api
Goal: reject oversize uploads without reading the whole body into memory (check `Content-Length`
and/or read at most `max_upload_bytes + 1`).
Acceptance criteria: GIVEN a body over the limit THEN 413 and at most `limit+1` bytes read; existing 413 test still passes; new test added.
Outcome:
- `caption_image` (`backend/app/api/routes.py`) now calls `image.read(max_upload_bytes + 1)` instead of reading the
  whole upload. A longer upload gets a 413.
- The 413 `detail` changed to "Image exceeds the {limit}-byte upload limit." because a bounded read can't report the
  true size. The status codes (200/400/413/415/422/503) and the `ErrorResponse` shape are unchanged.
- New test `test_captions_oversize_upload_reads_at_most_limit_plus_one` spies on `UploadFile.read`. Given a body
  10× the limit, it asserts a 413, no unbounded `read()`, at most `limit+1` bytes returned, and no predictor call. It
  failed on the old code (`unbounded read() call: [(-1, 10240)]`).
- New test `test_captions_accepts_upload_exactly_at_limit` pins the boundary: exactly `limit` bytes returns 200. The
  existing 413 test is unchanged and passes.
- Verification: backend 18 passed (TensorFlow not imported); full suite 96 passed; ruff lint + format clean; mypy 0
  issues (71 files); pre-commit hooks passed on both files.
- Committed as `cd5ee1c` (fix) and `b42fae6` (test).
- Residual, out of scope: Starlette still receives and spools the whole multipart body (in memory up to 1 MiB, then a
  temporary file) before the route runs. The bound applies to what the route loads, not to what the server accepts.
  See [`SECURITY.md`](SECURITY.md) § Known gaps.

### TASK-008 — Restore backend auto-deploy to the HF Space            [status: done] (deployed and verified 2026-10-03)
Area: deployment
Goal: no `deploy-backend.yml` run has succeeded since mid-June. The 2026-06-16/17 runs failed with HF HTTP 429
rate limits. The 2026-09-24 run was rejected as non-fast-forward because the Space's git history has commits that
aren't on GitHub `main`. Inspect the Space history, then either merge those commits into GitHub `main` or decide
that GitHub is the source of truth and force-push once.
Acceptance criteria: a green `deploy-backend.yml` run; Space `/healthz` reports `model_loaded: true`.
Findings: two faults.
- The Space's only extra commit, `302e907`, has a tree identical to GitHub's `64f80e8` (history rewritten after deploy).
- The Space has been in `CONFIG_ERROR` ("Missing configuration in README") because `befac80` removed the README
  YAML header.
Fix (ADR-017): `deploy-backend.yml` deploys the tested SHA, skips superseded commits, adds the header to a deploy
commit, force-pushes, and gates on HF runtime `RUNNING` + `/healthz` `model_loaded: true`. Manual runs must
also prove the exact SHA passed CI.
Outcome:
- Committed as `a2ee431`..`915112b`. CI run `37140358717` passed (all 6 jobs).
- `workflow_dispatch` run `37140993110` deployed `915112b` as Space commit `123c5aa`: built, then `RUNNING` in
  about 90 s. The workflow and an independent check both confirmed `/healthz` HTTP 200 with
  `model_loaded: true`, `model_version: v1.0.0`; `/docs` and `/openapi.json` returned HTTP 200.
- The Space's previous `CONFIG_ERROR` is resolved.
- Which Hub weights revision is loaded was resolved by TASK-004: `v2.0.0`, now also reported as `model_version`.

---

## Phase 3 — Multimodal baselines (decomposed below, NOT started)

Constraints already fixed: baselines live in the optional `[hf]` extra (`transformers==4.41.2`,
`torch==2.3.0`) and must not unpin the research pipeline (`tensorflow-cpu==2.15.0`). Every baseline
writes the standard `results/<run_id>/` artefact contract on the **same slice, reference count and
tokenisation** as the existing runs.

- [ ] **3A** — Side-by-side comparison harness: CNN+Transformer vs BLIP-base vs ViT-GPT2 vs GIT-base-coco
  → TASK-009, TASK-010, TASK-011, TASK-012
- [ ] **3B** — Per-model BLEU / CIDEr / METEOR / ROUGE-L on a shared COCO slice with deterministic tokenisation
  → TASK-013, TASK-014
- [ ] **3C** — Per-model latency benchmarking (single-image, batch, CPU vs GPU) → TASK-015, TASK-016
- [ ] **3D** — Comparison-result dashboard exposed through the existing SPA → TASK-007, TASK-017, TASK-018

Facts the tasks rely on (checked 2026-10-05):
- Slice: both committed runs (`stabilized-greedy`, `stabilized-beam-w4-lp07-rp12`) score the same 500 images in the
  same order, with about 1.46 references per image. References keep their `[start] … [end]` sentinels. Image paths
  are Kaggle paths (`/kaggle/input/datasets/awsaf49/coco-2017-dataset/…`).
- Tokenisation: references go through `preprocess_caption`, and metrics strip the sentinels
  (`evaluation/tokenization.py`). Baseline output must use the same normalisation; no second path.
- `RunMeta` has no latency or device fields, so 3C extends the artefact contract.
- `[hf]` is installed in the local venv. CI installs only the dev/eval requirements, so Phase 3 code imports
  `transformers`/`torch` lazily and its tests use fakes. mypy ignores missing `transformers.*` imports but not
  `torch.*`.
- The slice comes from COCO train2017. The HF baselines were, per their model cards, fine-tuned on COCO training
  data, so they have very likely seen these images; the CNN + Transformer held them out. This is disclosed, not fixed.
- Out of scope for Phase 3 (from `restructure-plan.md`, not in the README roadmap): `GET /v1/models`; a live
  `POST /v1/compare` on the Space, which would put `torch` in the image against ADR-013; and a `model-eval.yml` PR
  comment (that workflow has never existed).

Approvals needed before implementation:

| Approval | Needed by |
|---|---|
| None (built and tested offline with fakes) | TASK-009, 010, 011, 012, 013, 015, 017 |
| Read-only Hugging Face API lookups to pin revision SHAs | TASK-009 |
| Downloads of the three baseline checkpoints (about 0.7–1 GB each) | TASK-014, TASK-016 (optional TASK-011 smoke run) |
| Owner-run Kaggle CPU and GPU sessions with COCO 2017 | TASK-014, TASK-016 |
| `@playwright/test` and a Chromium download, locally and in CI | TASK-007, then TASK-018 |

Not needed: changes to `requirements.txt`, the `Dockerfile` or the Space; new runtime dependencies; any change to
the `tensorflow-cpu` pin.

Dependency graph (`*` owner-run, needs approvals; `†` blocked on install approval). First task: TASK-009.

```
TASK-009 (protocol)
 ├─► TASK-010 (slice loader) ─┬─► TASK-012 (runner) ─────────► TASK-014* (baseline runs) ─┐
 │                            ├─► TASK-013 (comparison check) ─► TASK-014*                 │
 │                            └─► TASK-015 (latency tool) ─► TASK-016* (CPU/GPU runs) ─────┤
 └─► TASK-011 (captioners) ───┬─► TASK-012                                                 ├─► TASK-017 (dashboard data) ─► TASK-018 (dashboard UI)
                              └─► TASK-015                                                 │                                   ▲
TASK-007† (Playwright E2E, start of 3D) ───────────────────────────────────────────────────┼───────────────────────────────────┘
```

TASK-007 can proceed in parallel with 3A–3C as soon as its install is approved.

### TASK-009 — Record the Phase 3 evaluation protocol before any baseline result            [status: done] (2026-10-05)
Area: evaluation · docs
Goal: fix the model list, slice, references, normalisation and decode settings in writing before any baseline runs,
so nothing can be tuned to the results.
Acceptance criteria:
- GIVEN `docs/EVAL_METHODOLOGY.md` THEN a Phase 3 section names each model with its Hub repo id, pinned revision SHA
  and licence. Presumed ids, to be confirmed: `Salesforce/blip-image-captioning-base`,
  `nlpconnect/vit-gpt2-image-captioning`, `microsoft/git-base-coco`; plus the CNN + Transformer at `v2.0.0`.
- The slice is the 500 images and references of `results/stabilized-greedy/predictions.jsonl`, in that order, with
  no re-sampling.
- Normalisation: baseline output goes through `preprocess_caption`, then `strip_sentinels`. The metric code is
  unchanged.
- Decode settings for each baseline, and whether the CNN + Transformer is compared greedy, beam or both, are fixed
  before any run.
- Each baseline's COCO training-data overlap is stated from its model card, with the caveat that scores aren't a
  held-out comparison.
- ADR-019 records where the baseline code lives; that it is imported lazily (`[hf]` stays optional; nothing in
  `backend/` or CI imports `transformers` or `torch`); and that the `tensorflow-cpu==2.15.0` pin is unchanged.
- This protocol is committed before any baseline results directory exists.
Verification: review; `git log -- docs/EVAL_METHODOLOGY.md results/` shows the protocol commit first; pre-commit on
the changed docs.
Depends on: none.
Owns: `docs/EVAL_METHODOLOGY.md` (new section), `docs/DECISIONS.md` (ADR-019), `docs/TASKS.md`.
Out of scope: code; 5-reference scoring of baselines; the latency protocol (TASK-015).
Outcome:
- Protocol: `EVAL_METHODOLOGY.md` § 8, committed as `ba71f38`. ADR-019 committed as `03493ee`. The `MEMORY.md`
  Phase 3 state was updated in `8870461`.
- Models and revisions (all four ids confirmed):
  - BLIP-base `Salesforce/blip-image-captioning-base` at `82a37760796d32b1411fe092ab5d4e227313294b`, BSD-3-Clause.
  - ViT-GPT2 `nlpconnect/vit-gpt2-image-captioning` at `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084`, Apache-2.0.
  - GIT-base-coco `microsoft/git-base-coco` at `a13141da42abd4a8cbf283601a8104265f537cee`, MIT.
  - CNN + Transformer: tag `v2.0.0` = `59d93b4babb16b0ac81eef598f3abc271a355cbf`, MIT.
- Sources: read-only Hub model API (`sha`, `cardData.license`, `license:` tag) and each model card at its pinned
  revision, plus the card of `ydshieh/vit-gpt2-coco-en-ckpts` (ViT-GPT2's cited upstream). No weights were
  downloaded and nothing was installed.
- Slice: `results/stabilized-greedy/predictions.jsonl`, 500 images, 732 references (1.46/image), file order. The
  beam run holds the same image list.
- Normalisation: `preprocess_caption` → `strip_sentinels`. Applied to all 1,000 committed CNN + Transformer
  predictions, it changes none.
- Decoding: the CNN + Transformer is compared with both runs, greedy as the primary comparison and beam
  (w4, lp 0.7, rp 1.2) as a labelled reference row. The baselines use greedy only: `num_beams` 1, no sampling,
  `max_new_tokens` 40, `repetition_penalty` 1.0, no prompt, float32, each model's own processor.
- Overlap caveat: the BLIP and GIT cards state COCO training. The ViT-GPT2 cards don't name a dataset, so it is
  treated as possibly COCO-trained. The CNN + Transformer held the slice out. Scores aren't presented as a
  held-out comparison, and the slice is kept on purpose.
- Before any baseline result:
  - `results/` is unchanged: `history.json` plus the two CNN + Transformer runs, last changed `c489b26` on
    2026-06-16.
  - `git log -- docs/EVAL_METHODOLOGY.md results/` lists `ba71f38` first.
- Checks: pre-commit passed on each changed doc. No code, config, results, README or workflow changes.

### TASK-010 — Load the evaluation slice from a committed run            [status: done] (2026-10-05)
Area: evaluation
Goal: one function returns the exact image list and references of an existing run, with image paths remapped to a
local images directory.
Acceptance criteria:
- GIVEN `results/stabilized-greedy/predictions.jsonl` and an images directory THEN it returns 500 entries in file
  order, references byte-identical, and paths of the form `images_dir/<basename>`.
- GIVEN the two committed runs THEN their slices compare equal (regression guard).
- No TensorFlow or `transformers` import, and no existence check on image files (the runner checks those).
Verification: `pytest tests/unit/test_eval_slice.py -q`; full suite; ruff; mypy; parity audit and notebook freeze
(`src/` change).
Depends on: TASK-009.
Owns: `src/captioning/evaluation/slice.py`, `tests/unit/test_eval_slice.py`, the `captioning.evaluation` export.
Out of scope: changing how `scripts/evaluate.py` builds its slice; downloading COCO.
Outcome:
- `load_eval_slice(predictions_path, images_dir)` returns a frozen `EvalSlice` (`image_paths`, `references`,
  `source`), exported from `captioning.evaluation`.
  - Rows stay in file order and references are kept exactly as stored.
  - Each image path becomes `images_dir/<file name>`; `/` and `\` separators are both handled.
  - Equality ignores `source`, so two runs' slices can be compared directly.
  - A row without a file name or a non-empty reference list raises `ValueError` naming the file and line.
- `tests/unit/test_eval_slice.py` (10 tests, offline, no COCO images):
  - the greedy slice matches the committed file: 500 images, 732 references, same order;
  - the greedy and beam slices compare equal;
  - no image files are needed;
  - Windows-style paths are remapped;
  - malformed rows are rejected;
  - a subprocess import loads no `tensorflow`, `transformers` or `torch`.
- Verification:
  - Focused 10 passed; full suite 114 passed.
  - ruff lint and format clean; mypy 0 issues (72 files).
  - Parity audit 4/4; notebook freeze OK; pre-commit passed.
  - `results/`, `scripts/evaluate.py` and the metric code are unchanged.
- Committed as `c6f9d7e` (loader) and `99f3dfb` (test).

### TASK-011 — Add a common captioner interface with Hugging Face adapters behind `[hf]`            [status: done] (2026-10-05)
Area: ml-core · evaluation
Goal: every compared model captions images through one interface, so 3B scores and 3C times them through the same
code.
Acceptance criteria:
- The interface takes a batch of image paths and returns normalised captions, plus the model's identity (model id,
  Hub id and revision, decode settings) for `run_meta.json`.
- The CNN + Transformer adapter wraps `CaptionPredictor` without changing it.
- One Hugging Face adapter, parametrised by Hub id and revision, covers the three baselines (or one adapter each if
  they can't share a loading path).
- Model ids, revisions and decode settings come from a new config section (`schema.py` + YAML, `extra="forbid"`),
  with defaults that leave parity unchanged.
- `transformers`/`torch` are imported only inside the adapter. Without `[hf]`, the package still imports, and a
  clear error names `pip install -e ".[hf]"`.
- Tests use fakes: no downloads and no `torch` import. Nothing changes in `backend/`, the `Dockerfile` or
  `requirements.txt`.
Verification: `pytest tests/unit/test_captioners.py -q`; a check that importing the package leaves `torch` out of
`sys.modules`; full suite; parity audit and notebook freeze; ruff; mypy (add `torch.*` to the mypy ignore list if it
is imported).
Depends on: TASK-009.
Owns: the new baseline package (location per ADR-019), `tests/unit/test_captioners.py`, the config schema section
and YAML, the mypy override, the `CLAUDE.md` layout line and `.claude/context/repo-map.md`.
Out of scope: serving baselines; fine-tuning; changing the CNN + Transformer's decoding. A real one-image smoke run
is optional and needs download approval.
Outcome:
- Config (`8f389e0`):
  - New strict `compare` section (`CompareConfig`, `ComparedModelConfig`, `BaselineDecodeConfig`, all
    `extra="forbid"`). Revisions must be full 40-character SHAs, and model ids must be unique.
  - Defaults are the § 8.1 / § 8.4 values, stated explicitly in `configs/base.yaml`. `serve` and the notebook
    parameters are unchanged.
- Package `captioning.baselines` (`a5e199e`), per ADR-019:
  - `Captioner`: a batch of image paths in, normalised captions out, through the existing `preprocess_caption` →
    `strip_sentinels` in one place.
  - `CaptionerIdentity`: model id, Hub repo, revision, and read-only decode settings.
  - `CNNCaptioner`: wraps `CaptionPredictor` unchanged, one `predict_path` per image. Its `from_artifacts`
    imports TensorFlow lazily.
  - `HFCaptioner`:
    - loads `AutoImageProcessor`, `AutoTokenizer` and `AutoModelForVision2Seq` at the pinned revision, in
      float32;
    - calls `generate()` with the protocol settings passed explicitly, and no prompt;
    - imports `torch`/`transformers` through `importlib`, only in `load()`;
    - without `[hf]`, raises `MissingHFDependencyError` (an `ImportError`) naming `pip install -e ".[hf]"`.
  - No mypy override was needed.
- `tests/unit/test_captioners.py` (`1ac78e9`, 21 tests, fakes only), covering:
  - config pins, YAML parity and strict validation;
  - reuse of the existing normalisation;
  - CNN wrapping and greedy/beam decode settings, and `from_artifacts` delegation without TensorFlow;
  - HF identity and `generate()` settings, and the pinned revision and dtype via fake modules;
  - the missing-`[hf]` error;
  - clean subprocesses showing that importing the package loads no `torch`, `transformers` or `tensorflow`, and
    that it imports and fails clearly with both blocked.
- Verification:
  - Focused 21 passed; full suite 135 passed (1 existing warning).
  - ruff lint and format clean; mypy 0 issues (76 files).
  - Parity audit 4/4; notebook freeze OK; pre-commit passed.
- Layout metadata (`ebe4ea3`): the repository layout line and `repo-map.md` list `baselines`.
- Not run, by design: real baseline inference, checkpoint downloads, a Hugging Face smoke test. TASK-014 is the
  first to exercise the real loading path.

### TASK-012 — Add the comparison runner that writes one results directory per model            [status: todo]
Area: evaluation
Goal: `scripts/compare_models.py` captions the slice with each selected model and writes `results/<run_id>/`
through the existing `write_run_artifacts`.
Acceptance criteria:
- Each model gets one directory containing the five standard files. Image order and references match the slice, and
  metrics come from the unchanged `compute_all_metrics`.
- `run_meta.json` records the model id, Hub id and revision, decode settings, sample count and max length.
- The runner refuses to overwrite an existing directory, checks every image exists before loading any model, and
  seeds all random generators.
- An end-to-end test with a fake captioner and a 3-image fixture produces valid files offline.
Verification: `pytest tests/unit/test_compare_models.py -q`; `python -m scripts.compare_models --help`; full suite;
ruff; mypy. If a Make target is added: `mingw32-make -n compare` and `tests/unit/test_makefile.py`.
Depends on: TASK-010, TASK-011.
Owns: `scripts/compare_models.py`, `tests/unit/test_compare_models.py`, an optional Make target.
Out of scope: real runs (TASK-014); the cross-model table (TASK-013).

### TASK-013 — Build the cross-model comparison summary with a slice-identity check            [status: todo]
Area: evaluation
Goal: join per-model results directories into one table, refusing to compare runs whose slice or references differ.
Acceptance criteria:
- GIVEN directories with identical image lists and references THEN it writes JSON and Markdown with each model's
  BLEU-1..4, METEOR, ROUGE-L, CIDEr, sample count, decode settings and run id. Values are copied verbatim from
  `metrics.json`.
- GIVEN any mismatch (image set, order, references, sample count) THEN it exits non-zero, names the differing run,
  and writes nothing.
- The output states the slice (500 images, about 1.46 references per image) and the overlap caveat from TASK-009.
- It passes on the two committed CNN + Transformer runs.
Verification: `pytest tests/unit/test_compare_runs.py -q`; a run over the two committed runs; full suite; ruff; mypy.
Depends on: TASK-009, TASK-010.
Owns: the comparison module or script and `tests/unit/test_compare_runs.py`.
Out of scope: running models; latency; the dashboard.

### TASK-014 — Run the baselines on the shared slice and publish the 3B results            [status: todo] (owner-run)
Area: evaluation · docs
Goal: produce and commit the per-model results directories and the comparison summary.
Acceptance criteria:
- GIVEN Kaggle with the same COCO 2017 dataset as the existing runs THEN there is one new `results/<run_id>/` per
  baseline.
- A CNN + Transformer greedy run through the harness reproduces the committed `stabilized-greedy` predictions; any
  differences are listed, not hidden.
- TASK-013's check passes over all runs, and the summary is committed.
- `README.md` and `EVAL_METHODOLOGY.md` cite the exact run ids, state the overlap caveat, and make no claims of
  superiority beyond the stated setup.
- Existing `results/*` stays untouched, and README edits are limited to its results section.
Verification: the Kaggle logs; `git diff --stat` shows only new results directories and docs; the comparison check
exits 0; pre-commit.
Depends on: TASK-012, TASK-013; approvals for the checkpoint downloads and the Kaggle session.
Owns: the new results directories, the comparison summary, the README and `EVAL_METHODOLOGY.md` results sections.
Out of scope: 5-reference rescoring; fine-tuning; retraining.

### TASK-015 — Add a latency benchmark for all compared models            [status: todo]
Area: evaluation
Goal: measure per-model caption latency for single images and batches on a named device, through the TASK-011
interface, and save it as a run artefact.
Acceptance criteria:
- Writes `results/<run_id>/latency.json` plus metadata: model and revision, device, batch sizes,
  TensorFlow/`torch`/`transformers` versions, warmup and repeat counts, and the reported statistics (chosen and
  documented in this task, before any run).
- Warmup is excluded, a monotonic clock is used, model load time is recorded separately, and every model gets the
  same inputs (the first N slice images).
- An offline test with a fake captioner and an injected clock checks the statistics and file shape.
- The addition to the artefact contract is documented in `EVAL_METHODOLOGY.md` and a `DECISIONS.md` entry.
Verification: `pytest tests/unit/test_latency_benchmark.py -q`; `--help`; full suite; ruff; mypy.
Depends on: TASK-010, TASK-011.
Owns: the benchmark script or module, `tests/unit/test_latency_benchmark.py`, the methodology section.
Out of scope: Space serving latency (`PredictorService` already reports it); Prometheus (Phase 4B); load testing.

### TASK-016 — Run CPU and GPU latency benchmarks and commit them            [status: todo] (owner-run)
Area: evaluation · docs
Goal: commit latency results for all four models on CPU and GPU.
Acceptance criteria:
- A named CPU environment and a Kaggle GPU each produce a single-image and a batch run per model, committed as new
  directories.
- The CNN + Transformer on GPU uses `tensorflow==2.15.0` (the GPU build of the same version) in that Kaggle
  environment only. The repository pin stays `tensorflow-cpu==2.15.0`.
- No cross-device claims beyond the measured setups.
Verification: the Kaggle logs; only new results directories in the diff; pre-commit.
Depends on: TASK-015; approvals for the Kaggle GPU session and the checkpoint downloads.
Owns: the new latency results directories.
Out of scope: latency on the HF Space.

### TASK-007 — Committed browser E2E for the caption flow            [status: blocked] (first task of Phase 3D; awaiting approval to install)
Area: frontend · deployment
Goal: promote the manual browser check in `TEST_PLAN.md` into a committed Playwright spec that mocks
`/healthz` and `/v1/captions` with `page.route` (no backend, no TensorFlow), run in the CI `frontend` job.
Acceptance criteria: GIVEN a mocked healthy API WHEN a PNG is uploaded and Generate clicked THEN the caption card
renders; GIVEN a disallowed file THEN an inline error shows and no request is sent; GIVEN the API is unreachable THEN
"Cannot reach backend" shows; zero console errors in each case.
Needs: `@playwright/test` devDependency + Chromium download (local and CI).
Depends on: approval to install `@playwright/test` and download Chromium, locally and in CI.
Owns: the Playwright config, `frontend/e2e/`, the `package.json` devDependency, the `ci.yml` frontend step, and the
`TEST_PLAN.md` frontend row.
Out of scope: dashboard specs (TASK-018).

### TASK-017 — Export dashboard data from committed results            [status: todo]
Area: evaluation · frontend
Goal: one script turns the 3B summary and the 3C latency results into a single static JSON file that the SPA
imports at build time.
Acceptance criteria:
- The JSON lists, per model: display name, Hub id and revision, metrics, latency per device and batch size, source
  run ids, the slice description and the overlap caveat.
- A test regenerates the JSON from the committed results and fails if the committed file has drifted.
- An ADR records the choice: static data in the SPA, with no backend endpoint and no live comparison. This keeps
  ADR-013 intact, since the Space image has neither `torch` nor `results/`.
Verification: the exporter's pytest; `npm run lint` and `npm run build`; full suite; ruff; mypy.
Depends on: TASK-013, TASK-015 (formats); the real data needs TASK-014 and TASK-016.
Owns: the export script, its test, the generated JSON (the frontend lane only reads it), the ADR.
Out of scope: backend endpoints; API contract changes.

### TASK-018 — Add the comparison dashboard to the SPA            [status: todo]
Area: frontend
Goal: a view in the existing SPA that renders the exported metrics and latency tables, every value traceable to its
run id, with the caveats visible.
Acceptance criteria:
- All models' metrics and their latency per device and batch size render; missing values show "n/a".
- The caption flow is unchanged, and the TASK-007 spec still passes.
- No new runtime dependency (no router or chart library without separate approval), and no network request is
  needed to render the dashboard.
- A Playwright spec covers the dashboard with zero console errors. Without TASK-007, it is checked manually and
  reported as not end-to-end verified.
- `TEST_PLAN.md` is updated.
Verification: `npm run lint`; `npm run build`; `npx playwright test`.
Depends on: TASK-017, TASK-007.
Owns: the dashboard components, the view switch in `App.jsx`, the dashboard spec, the `TEST_PLAN.md` frontend row.
Out of scope: live per-image comparison; a gallery of slice images; backend changes.
