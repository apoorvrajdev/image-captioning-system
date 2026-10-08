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

## Phase 3 — Multimodal baselines (done 2026-10-07)

Constraints already fixed: baselines live in the optional `[hf]` extra (`transformers==4.41.2`,
`torch==2.3.0`) and must not unpin the research pipeline (`tensorflow-cpu==2.15.0`). Every baseline
writes the standard `results/<run_id>/` artefact contract on the **same slice, reference count and
tokenisation** as the existing runs.

- [x] **3A** — Side-by-side comparison harness: CNN+Transformer vs BLIP-base vs ViT-GPT2 vs GIT-base-coco
  → TASK-009, TASK-010, TASK-011, TASK-012
- [x] **3B** — Per-model BLEU / CIDEr / METEOR / ROUGE-L on a shared COCO slice with deterministic tokenisation
  → TASK-013, TASK-014
- [x] **3C** — Per-model latency benchmarking (single-image, batch, CPU vs GPU) → TASK-015, TASK-016
- [x] **3D** — Comparison-result dashboard exposed through the existing SPA → TASK-007, TASK-017, TASK-018
  (TASK-007 added the Playwright E2E, covering the caption flow and the dashboard, on 2026-10-07)

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
| `@playwright/test` and a Chromium download, locally and in CI | TASK-007, then TASK-018 (granted 2026-10-07) |

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

### TASK-012 — Add the comparison runner that writes one results directory per model            [status: done] (2026-10-06)
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
Outcome:
- `python -m scripts.compare_models --config … --images-dir … --model <id> [--model …]` (`c7d71f9`):
  - Loads the slice with `load_eval_slice` (default `results/stabilized-greedy/predictions.jsonl`).
  - Builds each selected model from `config.compare` through the TASK-011 adapters, captions the slice in file
    order (`--batch-size`, default 1), and scores it with the unchanged `compute_all_metrics`.
  - Writes a new `results/<prefix><model_id>-<decoding>/` (default prefix `phase3-`) per model, through
    `write_run_artifacts`.
- `run_meta.json` follows § 8.6: for baselines, `weights_path` and `tokenizer_dir` are `<hub repo>@<revision>`,
  `max_length` is 40, and `decode_strategy` is greedy.
- Each run directory also gets `comparison_meta.json`, recording:
  - the protocol reference, backend and captioner class, Hub repo, revision, and full decode settings;
  - the normalisation path;
  - the slice source, content fingerprint (SHA-256 of image file names plus references, independent of line
    endings and the images directory), image count and reference count;
  - batch size, device, seed, and which metrics ran.
- Before any model loads, the runner fails if:
  - the slice isn't 500 images and 732 references (`--expected-*`);
  - a model id is unknown or selected twice;
  - the baseline decode settings sample;
  - the CNN lacks `--cnn-weights` / `--cnn-tokenizer-dir`;
  - a run directory already exists (never overwritten);
  - any slice image is missing.
- Per model, it also fails if the captioner's identity doesn't match the config.
- Seeds come from `set_global_seed(config.train.seed)`, as in `scripts/evaluate.py`. `torch` isn't seeded here
  (ADR-019 keeps it out of the runner), and the protocol decoding draws no random numbers.
- `tests/unit/test_compare_models.py` (`445c183`, 11 tests, fakes on the real `HFCaptioner` identity, a 3-image
  fixture, offline). It covers:
  - one contract directory per model, with order, references, normalisation, `run_meta.json` and
    `comparison_meta.json` checked;
  - batching, and isolation between models;
  - the committed slice as the default, with its pinned fingerprint `6b5628bf…`, matching the beam run;
  - each pre-run failure;
  - config propagation;
  - that importing the runner loads no `torch`, `transformers` or `tensorflow`.
- Verification:
  - Focused 11 passed; full suite 146 passed (1 existing warning).
  - ruff lint and format clean; mypy 0 issues (77 files); `--help` exits 0; pre-commit passed.
  - No `src/` or `configs/` change, so the parity audit and freeze don't apply.
- Not run, by design: any real model; TASK-014 is the first real run. No Make target was added.

### TASK-013 — Build the cross-model comparison summary with a slice-identity check            [status: done] (2026-10-06)
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
Outcome:
- `captioning.evaluation.comparison` (`7e5f5b5`) provides `load_run` and `build_summary`.
  - The slice fingerprint is recomputed from each run's `predictions.jsonl`; `slice_fingerprint` moved into the slice
    module (`7885850`, no behaviour change).
  - It's checked against the run's `comparison_meta.json` (fingerprint, image and reference counts, model id) and
    against `run_meta.json` `n_samples` and `metrics.json` `n_examples`.
  - All runs must then share image count, reference count and fingerprint. Runner outputs must also share protocol and
    normalisation, and no two runs may share a model and decoding.
  - Any failure raises a `ComparisonError` naming the run and the mismatch. Nothing is repaired.
- `python -m scripts.compare_runs RUN_DIR... [--reference-run DIR] --output-dir DIR` (`f1a5ab7`) writes
  `comparison.json` and `comparison.md`.
  - `comparison.json` holds exact metric values copied from `metrics.json`; `comparison.md` is the two-decimal table.
  - Rows are sorted by model id, decoding and run id. The output has no timestamps.
  - The output states the slice (counts, about 1.46 references per image, fingerprint) and the § 8.5 overlap caveat.
  - It refuses an existing output directory, and writes nothing if any check fails.
- Positional runs must have `comparison_meta.json`. The committed pre-harness runs (§ 8.4's CNN + Transformer rows)
  are accepted only through `--reference-run`.
  - They're labelled `reference`, with Hub repo and revision left null rather than inferred.
  - They're checked by the same fingerprint.
- `tests/unit/test_compare_runs.py` (`7d2b228`, 18 tests) uses run directories written by the real TASK-012 runner
  with its fake captioner. It covers:
  - the summary contents and verbatim metrics;
  - independence from argument order;
  - the committed reference runs, with pinned fingerprint and counts;
  - rejection of a changed fingerprint, counts, protocol, normalisation, references, slice, or a missing or
    malformed file;
  - duplicates;
  - the output-collision policy.
- Verification:
  - Focused 18 passed; full suite 164 passed (1 existing warning).
  - ruff lint and format clean; mypy 0 issues (79 files); parity audit 4/4; notebook freeze OK; pre-commit passed.
  - The run over the two committed runs passes (500 images, 732 references, fingerprint `6b5628bf…`).

### TASK-014 — Run the baselines on the shared slice and publish the 3B results            [status: done] (2026-10-06)
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
Outcome:
- Runs: one `scripts/compare_models.py --config configs/base.yaml --images-dir data/coco2017/train2017 --model <id>`
  invocation per model, with no protocol change.

  | Run directory | Revision | Commit |
  |---|---|---|
  | `results/phase3-blip-base-greedy/` | `82a37760796d32b1411fe092ab5d4e227313294b` | `cc71c5f` |
  | `results/phase3-vit-gpt2-greedy/` | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | `2a5ee2b` |
  | `results/phase3-git-base-coco-greedy/` | `a13141da42abd4a8cbf283601a8104265f537cee` | `e7c82c4` |
  | `results/phase3-inceptionv3-transformer-stabilized-greedy/` (CNN + Transformer reproduction) | `59d93b4` (tag `v2.0.0`) | `a1ab352` |

- What each run's `comparison_meta.json` records:
  - protocol § 8 and normalisation `preprocess_caption -> strip_sentinels`;
  - slice fingerprint `6b5628bf…`, 500 images, 732 references;
  - the § 8.4 decode settings, float32, batch size 1, device `cpu`, seed 42.
- Each baseline loaded at its pinned revision: the loaded config's `_commit_hash` equalled the § 8.1 SHA.
- CNN + Transformer reproduction:
  - Matches `results/stabilized-greedy/` exactly: 500/500 identical predictions, and all seven metrics bit-identical.
  - The weights were Hub commit `59d93b4`, with `model.h5` SHA-256 `74963a3f…`, materialised under
    `outputs/hub/…@59d93b4…/`. The local `models/v1.0.0/` is the development scaffold and wasn't used.
- Summary (`8993343`):
  - `results/phase3-comparison/` was written by `scripts/compare_runs.py` over the three baseline runs, with
    `results/stabilized-greedy/` and `results/stabilized-beam-w4-lp07-rp12/` as reference rows (§ 8.4).
  - The check passes over all five runs, and the output is byte-identical whatever order the runs are given in.
  - Passing the reproduction run alongside `stabilized-greedy` is rejected as a duplicate, as designed. The
    reproduction run is therefore kept as the record of the check, not as a comparison row.
- Docs (`ba888b3`): `EVAL_METHODOLOGY.md` § 8.8 records the run ids and revisions, the execution, the reproduction and
  the summary table, under the § 8.5 overlap caveat. It makes no held-out or superiority claim.
- Execution: on a local CPU instead of a Kaggle session (§§ 8.1–8.6 don't fix the host).
  - Images: the 500 slice files were fetched by file name, through the Kaggle API, from the dataset the committed runs
    read (`awsaf49/coco-2017-dataset`). All 500 decode; the image-bytes digest is in § 8.8.
  - Metric environment: rescoring the committed greedy predictions locally reproduces their metrics exactly.
- Artefact normalisation:
  - pre-commit converted the new files to LF and appended the final newline that `write_run_artifacts` omits from
    `metrics.json` and `run_meta.json`. The committed runs carry the same final newline.
  - No other byte changed: the Git blob hashes were checked before and after. The summary regenerates byte-identically.
- Not done: `README.md` doesn't cite the run ids yet. It holds owner-staged edits, so it was left untouched at the
  owner's request. Citing the results there is a follow-up.
- Verification:
  - Focused tests (`test_compare_models`, `test_compare_runs`, `test_eval_slice`, `test_captioners`): 60 passed.
  - Full suite: 164 passed (1 existing warning).
  - ruff lint and format clean; mypy 0 issues (79 files); notebook freeze OK.
  - pre-commit passed on every commit.
  - `git diff --stat 7d9e0f3` shows only additions: the new results directories and docs.
  - No `src/` or `configs/` change, so the parity audit doesn't apply.

### TASK-015 — Add a latency benchmark for all compared models            [status: done] (2026-10-07)
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
Outcome:
- `captioning.evaluation.latency` (`b191f4a`) imports no TensorFlow, `torch` or `transformers`. It provides:
  - `LatencySettings`: N images, batch sizes, warmup passes and measured passes. It rejects:
    - N below 1;
    - an empty, unsorted or repeated list of batch sizes;
    - a batch size that doesn't divide N;
    - zero warmup passes or zero measured passes.
  - `time_load`: times captioner construction plus `load()` once.
  - `measure_latency`: for each batch size, in ascending order, runs untimed warmup passes, then measured passes that
    time every `Captioner.caption()` call with `time.perf_counter`. A failed call, a wrong caption count or a clock
    going backwards ends the run.
  - `summarize`: count, mean, median, min and max.
  - `runtime_info`: Python, platform, and the installed `tensorflow`, `tensorflow-cpu`, `torch` and `transformers`
    versions.
- `python -m scripts.benchmark_latency --config … --images-dir … --model <id> --device cpu|cuda --environment "…"`
  (`743e2df`):
  - Selects, builds and identity-checks the model through TASK-012's `resolve_models`, `build_captioner` and
    `_check_identity`. One model per invocation.
  - Defaults (§ 9): the first 32 slice images, batch sizes 1 and 8, 1 warmup pass, 5 measured passes.
  - Fails before any model loads on:
    - invalid settings;
    - a slice that isn't 500 images and 732 references;
    - `--num-images` beyond the slice;
    - an unknown model, or a CNN without its checkpoint;
    - an existing run directory;
    - a missing benchmark image.
  - For the CNN + Transformer, `--device` is checked against the GPUs TensorFlow can see.
  - Writes only `results/phase3-latency-<model_id>-<decoding>-<device>/latency.json`. It has no timestamps, and the
    script never overwrites an existing run.
- `tests/unit/test_latency_benchmark.py` (`2d30ea6`): 42 tests, offline. They use a scripted clock and fake captioners
  (the real HF identity, and the real `CNNCaptioner` around a fake predictor). They cover:
  - hand-computed statistics and settings validation;
  - warmup exclusion, the exact sample count and call order, and two clock reads per sample;
  - failures during warmup and measurement, a wrong caption count and a backwards clock;
  - load timing and the recorded runtime versions;
  - the full `latency.json` shape and model identity, and byte-identical output for identical timings;
  - device propagation, every pre-load failure and the collision refusal;
  - that only the first N images must exist, and that a failed run writes nothing;
  - the CNN device check, imports that load no model framework, and `--help`.

  A mutation that timed the warmup pass failed 4 of these tests.
- Docs:
  - `EVAL_METHODOLOGY.md` § 9 is the latency protocol, fixed before any run (`5b400bf`).
  - ADR-020 is in `6df893d`.
  - The repo map and the evaluation skill list the tooling (`dc831da`).
- Verification:
  - Focused 42 passed; full suite 206 passed (1 existing warning).
  - ruff lint and format clean (100 files); mypy 0 issues (81 files).
  - Parity audit 4/4; notebook freeze OK; pre-commit passed on every commit.
  - `git diff --stat 39e59b9` shows no change under `results/`, `configs/`, `notebooks/` or `models/`, and none to
    `README.md`.
- Not run, by design: any real model, checkpoint download or timing run. No `latency.json` exists yet; TASK-016 makes
  the first runs. No Make target was added.

### TASK-016 — Run CPU and GPU latency benchmarks and commit them            [status: done] (2026-10-07)
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
Outcome, CPU half:
- Runs: one `scripts/benchmark_latency.py --config configs/base.yaml --images-dir data/coco2017/train2017 --model <id>
  --device cpu --environment "<label>"` invocation per model, with the § 9 defaults and no protocol change. The CNN
  also took `--cnn-weights` and `--cnn-tokenizer-dir`.

  | Run directory | Revision | Commit |
  |---|---|---|
  | `results/phase3-latency-blip-base-greedy-cpu/` | `82a37760796d32b1411fe092ab5d4e227313294b` | `672b224` |
  | `results/phase3-latency-vit-gpt2-greedy-cpu/` | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | `0857e22` |
  | `results/phase3-latency-git-base-coco-greedy-cpu/` | `a13141da42abd4a8cbf283601a8104265f537cee` | `c5d700c` |
  | `results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cpu/` | `59d93b4` (tag `v2.0.0`) | `df64b28` |

- Named CPU environment: the owner's laptop, an ASUS TUF Gaming A15 with an AMD Ryzen 7 7435HS (8 cores, 16 threads),
  15.8 GiB RAM, Windows 11 (build 26200), on AC power with the Turbo plan.
  - Software: `torch` 2.3.0+cpu, `tensorflow-cpu` 2.15.0, `transformers` 4.41.2, Python 3.10.11.
  - The baselines loaded from the local Hugging Face cache at their pinned snapshots, with `HF_HUB_OFFLINE=1`.
  - The CNN used the § 8.8 checkpoint (`model.h5` SHA-256 `74963a3f…`).
  - The laptop's RTX 2050 wasn't used: TF 2.15 has no native-Windows GPU support, and `torch` is the CPU build.
- Inputs: the first 32 slice images (fingerprint `6b5628bf…`), with batch sizes 1 and 8, 1 warmup pass and 5 measured
  passes. All four runs completed, with no failed call and no retry.
- Checks on every committed `latency.json`:
  - the pinned identity and decode settings;
  - the same 32 file names, fingerprint, settings, timing definition, environment and versions;
  - 160 and 20 samples;
  - summaries that recompute exactly from the raw samples;
  - no timestamps.
- Statistics: `EVAL_METHODOLOGY.md` § 9.9 (`11e749c`), copied from the files.
  - The CNN + Transformer's batch-8 figures are eight sequential single-image calls (§ 9.5), not batched inference.
  - No ranking or cross-device claim is made.
- Artefact normalisation: pre-commit's `mixed-line-ending --fix=lf` converted the script-written CRLF files to LF. Each
  committed blob equals the Git blob hash recorded before the conversion.
- Verification:
  - `git diff --stat cca6a0d` shows only the four new `results/phase3-latency-*-cpu/` directories and docs. No
    quality run and no `phase3-comparison/` file changed.
  - pre-commit passed on every commit.
  - No code changed, so the test suite wasn't re-run.

Outcome, GPU half (same protocol, same 32 images):
- Kaggle: a dedicated private kernel, `apoorvujjwal/task-016-phase-3-gpu-latency-benchmark` (version 1).
  - It was pushed with the Kaggle CLI, with the `NvidiaTeslaT4` accelerator, internet on, and
    `awsaf49/coco-2017-dataset` attached.
  - No existing notebook was used or changed.
  - It ran for about 15 minutes and finished `COMPLETE`.
- Hardware: two Tesla T4s (15360 MiB, driver 580.178.04). Every run used GPU 0, through `CUDA_VISIBLE_DEVICES=0`. The
  host had an Intel Xeon @ 2.00GHz with 4 vCPUs and 31.3 GiB RAM.
- Runtime:
  - The Kaggle image is Python 3.13.15, and `tensorflow` 2.15.0 has no wheels for it. The kernel therefore used uv to
    build two Python 3.10.20 environments.
  - Both environments hold the `requirements.txt` pins without `tensorflow-cpu`, and the repository at `03a8f9e`.
  - Hugging Face models: `torch` 2.3.0+cu121 and `transformers` 4.41.2.
  - CNN: `tensorflow==2.15.0` plus its own `and-cuda` CUDA library pins, without the three TensorRT packages, which
    can't be installed from PyPI and aren't used by the CNN.
  - Two environments were needed because `torch` 2.3.0 and `tensorflow` 2.15.0 pin conflicting cuDNN and cuBLAS builds.
  - The repository pins are unchanged.
- Checks before any run:
  - `torch` ran a convolution on `cuda:0`. TensorFlow listed `GPU:0` (Tesla T4, compute capability 7.5) and ran a
    convolution and a matrix multiply on it.
  - The baselines loaded with `_commit_hash` equal to their pinned SHAs. The runs then used the cache with
    `HF_HUB_OFFLINE=1`.
  - The CNN's `model.h5` (`74963a3f…`) and `vocab.pkl` (`178029c9…`) matched the Hub commit `59d93b4` hashes.
- Runs: the four `--device cuda` CLI invocations ran in one session. All completed, with no failed call and no retry.

  | Run directory | Revision | Commit |
  |---|---|---|
  | `results/phase3-latency-blip-base-greedy-cuda/` | `82a37760796d32b1411fe092ab5d4e227313294b` | `4ba72c8` |
  | `results/phase3-latency-vit-gpt2-greedy-cuda/` | `dc68f91c06a1ba6f15268e5b9c13ae7a7c514084` | `b249763` |
  | `results/phase3-latency-git-base-coco-greedy-cuda/` | `a13141da42abd4a8cbf283601a8104265f537cee` | `2b70e88` |
  | `results/phase3-latency-inceptionv3-transformer-stabilized-greedy-cuda/` | `59d93b4` (tag `v2.0.0`) | `a072492` |

- Retrieval: `kaggle kernels output`. The files were committed unchanged (SHA-256 checked); they were already LF.
- Checks on the four files: the same checks as the CPU half, with `device` `cuda`. The CNN records
  `tensorflow_gpus: ["/physical_device:GPU:0"]`. Inputs, settings, timing and revisions all equal the CPU runs'.
- Setup note: the CNN's GPU `load_seconds` includes Keras downloading the 88 MB ImageNet InceptionV3 base weights,
  because the fresh machine had no Keras cache. The local CPU run had them cached. Samples aren't affected.
- Statistics: `EVAL_METHODOLOGY.md` § 9.10 (`04a190f`).
  - The CNN's batch-8 figures are sequential single-image calls.
  - No ranking is made, and no CPU-vs-GPU claim: the two halves differ in host, OS and framework builds.
- Verification:
  - `git diff --stat 03a8f9e` shows only the four new `results/phase3-latency-*-cuda/` directories and docs.
  - No quality run, CPU latency run or `phase3-comparison/` file changed.
  - pre-commit passed on every commit. No code changed.

### TASK-007 — Committed browser E2E for the caption flow            [status: done] (2026-10-07)
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
Outcome:
- Install, approved 2026-10-07:
  - `@playwright/test` ^1.63.0 is a devDependency (`e8896d2`). The lock adds only `@playwright/test`, `playwright` and
    `playwright-core` 1.63.0, which need Node >= 20; CI uses 20.
  - Chromium only: Chrome for Testing 153.0.8010.12, the full build locally and only the headless shell in CI.
  - The committed lock already failed `npm ci`: it had no entries for the `@emnapi/core` and `@emnapi/runtime`
    1.10.0 pinned by the optional `@rolldown/binding-wasm32-wasi`. It was resynced first, as its own commit
    (`a26e730`). Only optional wasm32 entries changed.
- Setup (`3595599`, ADR-023):
  - `frontend/playwright.config.js` builds the SPA and serves it with `npm run preview`, so the specs test the
    production bundle. One Chromium project, no retries; traces are kept on failure.
  - `frontend/e2e/support.js` has two auto-fixtures:
    - `api` answers every off-origin request. `/healthz` and `/v1/captions` get bodies shaped like
      `backend/app/schemas/caption.py`, or a refused connection when a test sets `api.down`. Any other off-origin
      request fails the test.
    - `consoleErrors` fails a test on any console error or uncaught page error. While `api.down` is set, Chromium's
      exact `Failed to load resource: net::ERR_CONNECTION_REFUSED` line is allowed for the two API URLs.
  - The test image is a 1×1 PNG built in memory, so no binary fixture is committed.
  - ESLint gets Node globals for these files, Playwright output is gitignored, and `npm run test:e2e` runs the suite.
- `e2e/caption-flow.spec.js` (`bafd4fa`) covers the criteria, 3 tests:
  - healthy API: PNG upload, then Generate, shows the caption card with version, strategy, latency and request ID;
  - a `.txt` file and a >10 MB file are each rejected inline, with no request;
  - API unreachable: the badge goes offline and "Cannot reach backend" shows.
- TASK-018's deferred dashboard spec landed with this task, as its own commit. TASK-007 had ruled it out of scope;
  the owner asked for it here. `e2e/phase3-dashboard.spec.js` (`366383e`) has 11 tests:
  - the view switch makes no request;
  - every quality and CPU/GPU latency cell equals the committed JSON, both the exact value and its display rounding;
  - provenance, the slice, the caveats (shown ahead of the tables) and every note;
  - "n/a" for the data's null revisions and decode settings;
  - the exact-values toggle;
  - the dashboard renders with the API down;
  - the caption state survives a view switch;
  - layout at 390 and 1280 px: no sideways page scroll, and no table cut off.
- CI (`1672472`): the `frontend` job, now "Frontend (lint + build + e2e)", installs the headless shell with its system
  dependencies, runs `npm run test:e2e`, and uploads traces only on failure.
- Docs: ADR-023 (`f793e4d`), `TEST_PLAN.md` (`334fa21`), the commands in `CI.md`, `CLAUDE.md`, the skills, the
  frontend lane and the repo map (`9e7ee7a`). The deployment skill's YAML check now reads files as UTF-8, because it
  failed on Windows (`4dec153`).
- Verification:
  - Local: 14 passed. Mutations were each caught by the right test, and every source file was then restored:
    - rounding metrics to 3 decimals → 2 tests failed;
    - a runtime fetch in the dashboard → the no-request test failed;
    - a `console.error` → all 14 failed;
    - removing the "Not a ranking" caveat → the caveat test failed.
  - `npm run lint` clean, `npm run build` OK, `npm ci` OK.
  - Full pytest 238 passed. Both workflows parse. Pre-commit passed on every changed file.
  - CI run `37645312836` on `4dec153`: all 6 jobs green. The frontend job ran 14 passed (10.1 s) on Linux.
  - Vercel deployed. The automatic backend deploy `37645500471` succeeded: Space `RUNNING`, healthy, `v2.0.0`.
  - No change under `frontend/src/`, `results/`, `notebooks/`, `models/`, `backend/`, `configs/` or `src/`, and none
    to `README.md`.
- Not done, by scope:
  - Other browsers.
  - JS unit tests.
  - A test that a missing metric or latency renders "n/a": the real data has none, and the specs use only the
    committed file.
  - Asserting that every column fits at 1280 px, which depends on the platform's fonts (slack 5.6–10.9 % with Windows
    fonts).

### TASK-017 — Export dashboard data from committed results            [status: done] (2026-10-07)
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
Outcome:
- `captioning.evaluation.dashboard` (`36f3a96`) and `python -m scripts.export_dashboard_data` (`fd2832c`) write
  `frontend/src/generated/phase3-dashboard.json` (`6e3791c`). The sources are `results/phase3-comparison/comparison.json`
  and every `results/phase3-latency-*/latency.json`. Nothing imports TensorFlow, `torch` or `transformers`.
- Contents: 4 models from 13 source runs (the 5 summary rows and the 8 latency runs).
  - Per model, sorted by model id (not a ranking): display name, backend, Hub id and revision, source run ids.
    - Quality rows: run id, kind, revision, decoding and settings, sample count, the seven metrics.
    - Latency per device and batch size: run id, environment, batch mode, load time, the § 9.4 summary.
  - Shared: the slice description (500 images, 732 references, fingerprint `6b5628bf…`), the § 8.5 overlap caveat
    copied from `comparison.json`, and notes restating §§ 8.4, 8.8, 9.1, 9.4, 9.5 and 9.10.
  - Values are copied verbatim. Parsing is strict, an int stays an int, and nothing is rounded, recomputed or
    converted. Raw latency samples stay in the run directories.
- The exporter refuses:
  - a missing or malformed source;
  - latency runs whose protocol, inputs, settings, timing or seed differ, or whose slice isn't the quality slice;
  - a run directory whose name doesn't match its file;
  - a summary that doesn't recompute from its samples, or a sample count that isn't calls × passes;
  - runs that disagree on a model's backend, Hub id or revision;
  - a summary row without its run directory, and a model without a display name.
- Provenance:
  - The CNN + Transformer's revision (`59d93b4`) comes from its latency runs.
  - Its two quality rows predate the harness and record no revision, so theirs is `null`. A note cites the § 8.8
    exact reproduction of the greedy row.
  - Its latency runs are marked `batch_mode: sequential` (§ 9.5).
- Location: `frontend/src/data/` was ruled out because the `.gitignore` dataset rule `data/` matches it. Prettier's
  pre-commit hook excludes `frontend/src/generated/` (`2137624`), because it reflows short arrays and exponents and
  the drift test owns the bytes.
- Tests (`579a038`): `tests/unit/test_dashboard_export.py`, 32 offline tests.
  - Drift: a fresh export equals the committed file. Every committed value is also checked against its raw source.
  - 17 refusal cases, each one edit to a copy of the real results.
  - CLI: `--check`, LF UTF-8 output, nothing written on error, `--help`. No model framework is imported.
  - Changing one committed value failed 5 tests; reverting the strict parsing failed 2.
- Docs: ADR-021 (`a98e707`); the repo map, the evaluation and frontend skills, and `TEST_PLAN.md` (`313ae79`).
- Verification:
  - Focused 32 passed; full suite 238 passed (1 existing warning).
  - ruff lint and format clean (103 files); mypy 0 issues (83 files); parity audit 4/4; notebook freeze OK.
  - pre-commit passed on every commit, run on the task files (README holds owner-staged edits).
  - `npm run lint` clean and `npm run build` OK. Nothing imports the JSON yet (TASK-018). A throwaway Vite 8 build
    outside the repo imported and bundled it.
  - `git diff --stat 843e15d` shows no change under `results/`, `configs/`, `notebooks/`, `models/` or `backend/`,
    none to existing frontend code, and none to `README.md`.
- Not done, by scope: any UI (TASK-018), a Make target, or a CI step beyond the pytest drift test.

### TASK-018 — Add the comparison dashboard to the SPA            [status: done] (2026-10-07; end-to-end spec added with TASK-007)
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
Outcome:
- `frontend/src/components/Phase3Dashboard.jsx` (`28af9f0`) renders `src/generated/phase3-dashboard.json`, imported at
  build time. The view switch in `App.jsx` (`4b6e636`) is two `aria-pressed` buttons, "Caption an image" and
  "Phase 3 comparison", with no router.
  - The caption flow stays mounted behind `hidden`, so its file, result and in-flight request survive a switch.
  - No dependency was added; `package.json` and `package-lock.json` are unchanged.
- What the dashboard shows, in order:
  - Caveats first: not live, not held-out (the § 8.5 caveat verbatim), not a ranking, CPU/GPU from different hosts,
    and sequential CNN batches.
  - The quality table: 5 runs, each with run id and kind, decoding, samples and the 7 metrics.
  - A CPU table and a GPU table: each model's run id, batch mode, load time, then per batch size the calls, the
    sample count and mean, median, min and max.
  - Per-model provenance cards: Hub repository linked at its revision, revision, source runs, and each quality run's
    revision and decode settings.
  - The slice facts, the latency settings and timing definitions, and every note in the file.
- Values are shown as exported (ADR-022). The dashboard ranks nothing, computes no winner and colours nothing by value.
  - Rounding follows the committed reports: metrics to 2 decimals (`comparison.md`), latency to 0.0001 s (§ 9.9's
    0.1 ms, in the file's unit, seconds), load time to 0.1 s.
  - Each number keeps its exact value in `<data value>`, and "Show exact values" displays it.
  - Missing values show "n/a".
- The generated JSON and its schema are unchanged (SHA-256 `cbb3b295…`). `python -m scripts.export_dashboard_data
  --check` reports it up to date, and `tests/unit/test_dashboard_export.py` passes (32).
- Docs: ADR-022 (`015218a`), `TEST_PLAN.md` dashboard checks (`4e7049c`), the repo map and frontend skill (`5f5138f`).
- Verification:
  - `npm run lint` clean; `npm run build` OK (26 modules; JS bundle 205 → 233 kB, 72 kB gzip). Pre-commit hooks passed
    on every changed file.
  - The Playwright spec isn't written: TASK-007 (the `@playwright/test` install) is still blocked on approval. Per
    the criteria, the dashboard was checked manually and is **not end-to-end verified**.
  - The manual check was a throwaway script, not committed. It drove the installed Chrome (headless, DevTools
    protocol) against `vite preview` of the production build, with `/healthz` and `/v1/captions` mocked the way
    TASK-007 plans. 40 of 40 checks passed:
    - every displayed metric equals `comparison.md`, and every latency row equals §§ 9.9–9.10;
    - every exact value is present in `<data value>`;
    - switching views makes no request, and the JSON is never fetched;
    - zero console errors or warnings;
    - the page doesn't scroll sideways at 390 px, and no table column is clipped at 1280 px;
    - the caption flow is unchanged: upload → Generate → card, a `.txt` rejected with no request, "Cannot reach
      backend" when the API is down, and the file and result survive a round trip to the dashboard.
  - A second throwaway build aliased the import to a mutated copy of the JSON, outside the repo. A deleted metric, a
    missing GPU run, a missing batch and a null load time each rendered "n/a", with no errors.
  - No change under `results/`, `configs/`, `notebooks/`, `models/`, `backend/` or `src/`, and none to `README.md`.
- Not done, by scope: the Playwright dashboard spec (needs TASK-007), a URL for each view, and the README Phase 3
  results (README holds owner-staged edits).
- Addendum, 2026-10-07: TASK-007 added the Playwright dashboard spec, `e2e/phase3-dashboard.spec.js` (`366383e`),
  with zero console errors. It runs in CI, so the dashboard is now end-to-end verified.

---

## Phase 4 — Production hardening and supply-chain reliability (done 2026-10-09)

Phase 4 hardens the system that already ships. It adds no features. Scope and rationale come from the Phase 4
reconnaissance, reviewed and approved on 2026-10-07. The facts below were re-checked against the repository the same
day. Work order: TASK-019 → TASK-020 → TASK-021 → TASK-022 → TASK-023.

**Deadline:** TASK-019 must land before **2026-10-19**, when GitHub moves `ubuntu-latest` to Ubuntu 26.

- [x] **4A** — CI platform currency → TASK-019 (done 2026-10-08)
- [x] **4B** — Serving dependency security → TASK-020 (done 2026-10-08)
- [x] **4C** — Dependency and secret scanning in CI → TASK-021 (done 2026-10-09)
- [x] **4D** — Deploy only when the production image changes → TASK-022 (done 2026-10-08)
- [x] **4E** — Real-model post-deploy smoke test → TASK-023 (done 2026-10-08)

Facts the tasks rely on (checked 2026-10-07):
- Runners: all seven jobs run on `ubuntu-latest`: five in `ci.yml`, one in `deploy-backend.yml` and one in
  `no-ai-attribution.yml`.
- Actions in use: `actions/checkout@v4`, `actions/setup-python@v5`, `actions/cache@v4`, `actions/setup-node@v4`,
  `actions/upload-artifact@v4`.
- Node: the `frontend` job sets `node-version: "20"`. Nothing else in the repository pins Node (no `.nvmrc`, no
  `engines`). Vite 8 needs `^20.19.0 || >=22.12.0`, and `@playwright/test` 1.63 needs Node >= 20.
- Python:
  - The pytest matrix is 3.10 and 3.11. `requires-python` is `>=3.10,<3.13`, ruff targets `py310`, mypy uses
    `python_version = "3.10"`, and the local venv is 3.10.11.
  - Python 3.10 reaches upstream end of life in October 2026.
  - `tensorflow-cpu` 2.15.0 publishes wheels for Python 3.9–3.11 only, so the matrix can't move to 3.12 while the TF
    pin holds.
- Serving pins (`requirements.txt`): `fastapi==0.111.0`, `python-multipart==0.0.9`, `pillow==10.3.0`,
  `uvicorn[standard]==0.30.1`. Starlette isn't pinned directly; FastAPI brings it in. `pyproject.toml` gives ranges
  for the same packages: `fastapi>=0.111,<1.0`, `python-multipart>=0.0.9`, `pillow>=10.0,<11.0`.
- Scanning: CI runs no dependency audit and no secret scan. The `gitleaks` pre-commit hook (v8.18.4) scans staged
  changes only, so it does nothing in CI (ADR-016). Both are open rows in `SECURITY.md` § Known gaps.
- Deploys:
  - `deploy-backend.yml` runs after every green CI run on `main` (`workflow_run`), so every commit force-pushes and
    rebuilds the Space, docs-only commits included (ADR-017). `workflow_run` doesn't support a path filter.
  - The image copies `requirements.txt`, `pyproject.toml`, `README.md`, `src/`, `backend/`, `configs/` and `models/`.
  - `README.md` is in the image on purpose. `pyproject.toml` declares `readme = "README.md"` for the in-image
    `pip install -e .`, and the deploy commit prepends the Space's config header to it.
  - The deploy gate checks HF `RUNNING` and `/healthz` `model_loaded: true`. It sends no caption request and checks no
    CORS header.
  - Spaces sleep when idle, and the first request after that is slow (runbook § 9).

Dependency graph. The work order above is the approved sequence; the arrows are hard dependencies.

```
TASK-019 (CI platform) ─┬─► TASK-020 (serving deps) ─► TASK-021 (scanning in CI)
                        └─► TASK-022 (deploy on image change) ─► TASK-023 (post-deploy smoke test)
```

### TASK-019 — CI platform currency            [status: done] (2026-10-08, deadline 2026-10-19)
Area: deployment
Goal: CI and the deploy workflow run on an explicitly chosen, supported platform before GitHub moves `ubuntu-latest`
to Ubuntu 26 and retires the Node 20 Actions runtime.
Scope:
- Replace `ubuntu-latest` in all seven jobs with an explicit supported runner, currently proposed as `ubuntu-24.04`.
- Update the GitHub Actions versions needed to clear the Node 20 deprecation. Confirm each exact version from the
  action's own releases during the task. This plan doesn't fix them.
- Move the `frontend` job from Node 20 to the supported LTS the repository standardises on, chosen and recorded in
  this task.
- Record the Python 3.10 end-of-life decision: whether 3.10 stays in the pytest matrix and `requires-python`, and why.
- Update `docs/CI.md`, and the deployment runbook wherever it names the runner or Node version.
Acceptance criteria:
- GIVEN the three workflow files THEN no job uses `ubuntu-latest`, and every job names the chosen runner.
- GIVEN a push to `main` THEN all six CI jobs pass on that runner, and the run shows no Node 20 deprecation annotation.
- The `frontend` job passes lint, build and the Playwright E2E on the new Node version.
- The next `deploy-backend.yml` run passes its existing gate (`RUNNING` + `model_loaded: true`) on the pinned runner.
- An ADR records the runner pin and its reason, the Node version, and the Python 3.10 decision.
- No gate is removed or weakened. The pytest matrix changes only if the recorded 3.10 decision says so.
Verification: the deployment skill's YAML parse check on all three workflows; `grep -rn "ubuntu-latest"
.github/workflows` finds nothing; pre-commit on the changed files; the CI run on `main` (every job green, annotations
checked); the following `deploy-backend.yml` run.
Depends on: none. First Phase 4 task.
Owns: `.github/workflows/{ci,deploy-backend,no-ai-attribution}.yml`, `docs/CI.md`, the runbook's CI notes, the ADR.
`pyproject.toml` only if the 3.10 decision changes `requires-python` or the tool targets.
Out of scope: serving dependency upgrades (TASK-020); vulnerability or secret scanners (TASK-021); redesigning the
deploy trigger (TASK-022); the TensorFlow / Keras migration; adopting Ubuntu 26; the Dockerfile base image; unrelated
application changes.
Outcome:
- Runner (`620881f`): all seven jobs name `ubuntu-24.04`, and `grep -rn "ubuntu-latest" .github/workflows` finds
  nothing.
- Actions (`98b9a56`): `actions/checkout@v7`, `actions/setup-python@v7`, `actions/setup-node@v7`, `actions/cache@v6`
  and `actions/upload-artifact@v7`. Each tag exists upstream and its `action.yml` declares `runs.using: node24`.
- Node (`aa11845`): the `frontend` job runs Node 24 (LTS). Nothing else pins Node.
- Python 3.10 stays in the pytest matrix and in `requires-python`, for the reasons in ADR-024. `pyproject.toml` is
  unchanged.
- Docs: ADR-024 (`3b463db`), `CI.md` § Platform (`1fe5b63`), and the deployment skill's DoD and verification commands
  (`bd91c3c`). The runbook names no runner or Node version, so it needed no change.
- Verification:
  - All three workflows parse, and pre-commit passed on the changed docs.
  - CI run `37799586272` on `bd91c3c`: all 6 jobs green on image `ubuntu-24.04` (20261004.327.1), with 0 annotations
    on every job, so no Node 20 deprecation notice.
  - Deploy run `37799794182` on the same image: Space `RUNNING`, then "Healthy: model_version=v2.0.0". The public
    `/healthz` returned `model_loaded: true`, `v2.0.0`.
- Not done, by scope: Ubuntu 26, SHA-pinned actions, the Dockerfile base image, the TensorFlow / Keras migration.

### TASK-020 — Fix vulnerable serving dependencies            [status: done] (2026-10-08)
Area: inference-api · deployment
Goal: the serving image ships no FastAPI, Starlette, `python-multipart` or Pillow version with a known, fixable
vulnerability, and the `tensorflow-cpu==2.15.0` pin stays.
Scope:
- Upgrade FastAPI, and with it Starlette, plus `python-multipart`, to releases that fix the known advisories. Pin
  Starlette explicitly if FastAPI's range would still allow a vulnerable version.
- Upgrade Pillow where appropriate: where a fixed release works with TF 2.15 and the current `<11.0` bound. If a fix
  needs a bound change, decide it in this task and record why.
- Keep the `requirements.txt` pins and the `pyproject.toml` ranges consistent.
- Reassess the full-body upload buffering gap (`SECURITY.md` § Known gaps, first row) against the upgraded multipart
  parser. Either close it or update the row with the current behaviour.
Acceptance criteria:
- An audit of `requirements.txt` (for example `pip-audit -r requirements.txt`) reports no known vulnerability in
  FastAPI, Starlette, `python-multipart` or Pillow. Any finding that remains is listed in `SECURITY.md` with its reason.
- `tensorflow-cpu==2.15.0` and `numpy<2` are unchanged, and the resolved set installs alongside them (`pip check` is
  clean).
- The existing backend contract tests pass unchanged. `/healthz` and `/v1/captions` keep the 200/400/413/415/422/503
  codes and the `CaptionResponse` / `ErrorResponse` shapes. No test is weakened.
- Any change to upload handling keeps that contract and adds a regression test.
- The Space redeploys on the new pins and passes the deploy gate.
Verification: `pytest backend/app/tests -q`; the full suite; ruff lint + format; mypy; `pip check`; the audit
command; pre-commit on the changed files; CI green on `main`; the deploy run.
Depends on: TASK-019.
Needs: approval to upgrade packages in the local venv and to install the audit tool.
Owns: `requirements.txt`, the `pyproject.toml` dependency ranges, `SECURITY.md`, any upload-handling fix in
`backend/app/` and its test.
Out of scope: TensorFlow, Keras or NumPy upgrades; CI scanners (TASK-021); dev, eval and `[hf]` dependencies;
frontend dependencies; API contract changes.
Outcome:
- Dependencies. `requirements.txt` pins and `pyproject.toml` ranges move together:
  - FastAPI 0.111.0 → 0.133.0, Starlette 0.37.2 → 1.3.1 and `python-multipart` 0.0.9 → 0.0.31 (`4e6a957`).
    - FastAPI 0.133.0 is the first release that admits Starlette 1.x. From 0.135.2 it would force Pydantic ≥ 2.9.
    - Starlette is now pinned explicitly, because FastAPI's own range (`>=0.40.0`) still admits vulnerable releases.
  - Pillow 10.3.0 → 12.3.0 (`c349eb0`). The range moves from `<11.0` to `>=12.3,<13.0`, because every fix is in 12.x.
  - `anyio` 4.4.0 → 4.14.2 (`8496066`), added with the owner's approval during review. anyio 4.4.0's idle worker
    threads kept a refused upload's spooled temporary file open. 4.14.2 fixes that, and fixes its two advisories.
  - `routes.py` uses Starlette's RFC 9110 status names (`6ed93d9`). The codes are the same, and the old names warn on
    every use.
- Buffering gap reassessed. The upgrade alone doesn't change it: on both the old and new pins, a 50 MB upload was read
  in full and spooled to disk before the 413. `BodySizeLimitMiddleware` (`6bc4347`) now caps a body at
  `max_upload_bytes` + 64 KiB:
  - with `Content-Length`, it's refused after 0 bytes;
  - chunked, it's refused at 10.6 MB;
  - under the cap the route's exact limit still decides: 10 MiB passes, 10 MiB + 1 byte gets the route's 413.
  - What's left (bodies up to the cap, bandwidth, the unconfigurable HF proxy) is a row in `SECURITY.md` § Known gaps.
- Tests (`a65546d`): five in `backend/app/tests/test_body_size_limit.py`, driven over raw ASGI so the bytes read can
  be counted. The existing contract tests are unchanged.
  - With the cap disabled, the declared and undeclared oversize tests fail.
  - On anyio 4.4.0, the temp-file test fails.
- Docs: ADR-025 (`bffae6b`), `SECURITY.md` § Dependency audit and the body-cap rows (`c8643af`), the README (`fd7b157`),
  and the test plan, skills and repo map (`e8c3eda`).
- Verification:
  - `pip-audit -r requirements.txt` (2.10.1): 89 findings in 7 packages before, 26 in 3 after (`keras`, `protobuf`,
    `click`), none in FastAPI, Starlette, `python-multipart`, Pillow or `anyio`. Each remaining finding is listed in
    `SECURITY.md` with its reason.
  - `pip check` is clean in the dev venv, and in a fresh `pip install -r requirements.txt` venv that mirrors the image
    layer. That venv also confirms `jinja2`, `fastapi-cli`, `orjson`, `ujson`, `email-validator` and `httpx` are no
    longer installed.
  - Locally: backend tests 23 passed; full suite 243 passed; ruff clean (105 files); mypy 0 errors (85 files); parity
    audit 4/4; notebook freeze OK; pre-commit passed on every changed file.
  - The generated OpenAPI changed in one place only: the upload field is `contentMediaType: application/octet-stream`
    instead of `format: binary`.
  - Real model on local uvicorn: 200, 400, 413 (declared, chunked and browser-like), 415 and 422 as before. CORS and
    `x-request-id` are on every response, `/docs` and `/openapi.json` return 200, and the log is clean.
  - `/code-review`: 7 findings.
    - A refused 413 resetting the connection is refuted: uvicorn discards the unread body and keeps the connection.
    - The temp file left open was confirmed and fixed by the anyio upgrade.
    - Two were covered in the tests: CORS on the 413, and the same 413 text from both layers.
    - Three were declined:
      - a per-route rather than global cap: `/v1/captions` is the only route that reads a body;
      - moving Pillow into `[hf]`: `[hf]` dependencies are out of scope here;
      - skipping the header scan for GET: it costs nothing measurable.
  - `/security-review`: no findings.
  - CI run `37806775992` on `e8c3eda`: all 6 jobs green, 243 passed on Python 3.10 and 3.11, 0 annotations. It
    installed the pinned versions.
  - Deploy run `37808007611`: the Space rebuilt on the new pins (deployment commit `52f48d6`), reached `RUNNING`, and
    reported "Healthy: model_version=v2.0.0". The public `/openapi.json` shows the new FastAPI, and `/v1/captions`
    still lists 200, 400, 413, 415, 422 and 503.
- Not done, by scope:
  - `keras` and `protobuf`, held back by the TF 2.15 pin;
  - `click`;
  - moving Pillow into the `[hf]` extra;
  - Starlette's `httpx2` notice for `TestClient` (dev dependency);
  - CI scanning (TASK-021);
  - a real caption request against production (TASK-023).

### TASK-021 — Dependency and secret scanning in CI            [status: done] (2026-10-09)
Area: deployment
Goal: CI catches known-vulnerable dependencies and committed secrets, instead of relying on hooks that only scan
staged changes on machines that have them installed.
Scope:
- `pip-audit` in CI over the serving requirements (`requirements.txt`). Whether other requirement files are audited is
  decided and recorded in this task.
- `npm audit` over the frontend's production dependencies (`--omit=dev`), which are what ships in the Vercel bundle.
- A full-history `gitleaks` scan in CI, with a full-history checkout, so a commit made without hooks is still scanned.
- `SECURITY.md` known-gap rows and `docs/CI.md` updated; an ADR records the scanners, what each covers and when each
  blocks.
Acceptance criteria:
- GIVEN a push or pull request THEN all three scans run in CI and their findings show in the job log.
- The scans become a blocking CI gate only once TASK-020's clean baseline is confirmed on `main`. If the baseline isn't
  clean, they land report-only, with the remaining findings listed.
- The full-history `gitleaks` scan passes over the whole history. A false positive gets a documented allowlist entry.
  A real secret is rotated first, never only allowlisted.
- Scanner versions are pinned. `gitleaks` matches the pre-commit hook's v8.18.4 unless the task records a reason to
  differ.
- Workflow permissions stay `contents: read`, and no new repository secret is needed.
Verification: the YAML parse check; a CI run on `main` showing the three scans; pre-commit on the changed files; the
same scan commands run locally where installation is approved.
Depends on: TASK-020 (clean baseline).
Needs: approval to add the scanner tooling to CI, and to install it locally to reproduce findings.
Owns: `.github/workflows/ci.yml` (the scan steps), any scanner config such as a `gitleaks` allowlist, `docs/CI.md`,
`SECURITY.md`, the ADR.
Out of scope: fixing vulnerabilities (TASK-020); container image scanning; Dependabot or automated update pull
requests; gating on dev-only npm dependencies.
Outcome:
- Implementation:
  - `ci.yml`: a new `security` job runs pip-audit 2.10.1, gated by `scripts/check_pip_audit.py`, then gitleaks 8.18.4
    over the full history. The `frontend` job ends with `npm audit --omit=dev` (npm 11.6.2). All three block.
  - `.github/pip-audit-baseline.txt`: TASK-020's 15 reviewed findings, each pinned to package, version and id, with its
    reason and removal condition.
  - `tests/unit/test_check_pip_audit.py`: 20 tests.
  - Docs: ADR-026, `SECURITY.md` § CI scanning policy and § Known gaps, `CI.md`, `TEST_PLAN.md`, the deployment skill,
    the repo map and `CLAUDE.md`.
- Acceptance clarifications (ADR-026):
  - **Blocking, not report-only.** TASK-020's residual findings are reviewed exceptions (ADR-025), so with them
    baselined the scan starts clean.
  - **Only `requirements.txt` is audited.** The dev and eval files aren't in the image. Their 43 findings (42 in
    `nltk` 3.8.1, 1 in `pytest` 8.2.2) are a known gap in `SECURITY.md`.
  - **gitleaks** matches the hook's v8.18.4. CI uses the release binary, SHA-256 checked, not `gitleaks-action`,
    which runs on Node 20 (ADR-024). No allowlist: the history is clean.
- Review: the security review before commit found that the gate read an audited dependency with no `vulns` list, or
  a non-list one, as clean. It now fails closed, and three regression tests fail without the fix. No other findings.
- Verification, locally on 2026-10-09:
  - 20 gate tests; ruff, ruff format and mypy clean; the three workflows parse; pre-commit passed on the changed
    files.
  - pip-audit 2.10.1 on `requirements.txt` (Windows, Python 3.10): 70 dependencies, 26 rows in 3 packages, 15
    distinct findings. The gate reports all 15 as `[baseline]`, with 0 new, 0 stale and 0 unaudited, and exits 0.
  - gitleaks 8.18.4: 201 commits scanned, the same count as `git rev-list HEAD --count`, and no leaks.
  - `npm audit --omit=dev` (npm 11.6.2): 0 vulnerabilities.
  - Negative checks, each exiting 1:
    - the gate on the pre-TASK-020 `requirements.txt`: 48 findings, 33 `[NEW]` (anyio 2, Pillow 17,
      python-multipart 7, Starlette 7);
    - gitleaks on a throwaway clone with a fake GitHub token committed and then deleted: found in history as
      `github-pat`, with the value `REDACTED` and absent from the log;
    - `npm audit --omit=dev` with `minimist` 1.2.5 as a production dependency. As a dev dependency it passes.
- Not done, by scope: container image scanning, gating dev and eval dependencies, Dependabot, and a scheduled scan.

### TASK-022 — Deploy only when the production image changes            [status: done] (2026-10-08)
Area: deployment
Goal: commits that can't change the production image (docs, tests, frontend, results and the like) no longer rebuild
and restart the HF Space.
Scope:
- In `deploy-backend.yml`, decide from the commits being deployed whether any image input changed, and skip the
  force-push and rebuild if none did. `workflow_run` has no path filter, so the check runs inside the job.
- Derive the image inputs from the `Dockerfile` `COPY` lines and `.dockerignore`, plus the `Dockerfile` itself and the
  deploy workflow, which writes the Space's README header.
- `README.md` is copied into the image on purpose (`pyproject.toml` `readme`, and the Space config header). Decide
  explicitly whether a README-only commit redeploys, and record why.
- A new ADR revising ADR-017 (`DECISIONS.md` is append-only).
Acceptance criteria:
- GIVEN a commit that changes no image input THEN the deploy run skips with a notice giving the reason, and nothing is
  pushed to the Space.
- GIVEN a commit that changes any image input THEN it deploys as it does today, through the same gate.
- The comparison base is the commit the Space last deployed, not the parent commit. An image change whose deploy was
  skipped, superseded or failed is still deployed by a later commit.
- `workflow_dispatch` can still force a deploy. The superseded-commit guard and the manual-run CI verification stay.
- `README.md` is handled as the ADR says.
- The ADR records the trigger rule and the README decision, and `docs/CI.md` and the runbook match it.
Verification: the YAML parse check; the change-detection rule exercised against representative commit ranges
(docs-only, image-changing, an image change followed by a docs-only commit); on `main`, one observed skip for a
docs-only push and one observed deploy for an image change.
Depends on: TASK-019 (same workflow file; the runner pin lands first).
Owns: `.github/workflows/deploy-backend.yml`, the ADR, `docs/CI.md`, the deployment runbook.
Out of scope: changing what the image contains; the CI workflow's own trigger; the post-deploy smoke test (TASK-023);
Vercel deploys (handled by Vercel's GitHub integration).
Outcome:
- Implementation:
  - `scripts/deploy_scope.py` (`b286300`) decides and records.
  - `deploy-backend.yml` (`36a7253`) runs the decision after the superseded-commit guard, gates every later step on it,
    and records the deploy last.
  - 67 tests (`b21b97a`).
  - Docs: ADR-027 (`d8ee2ce`), `CI.md`, the runbook and `SECURITY.md` (`8fd05f1`), and the skill, test plan and repo map
    (`fbf55d8`).
- Image inputs: the Dockerfile's `COPY` sources (`requirements.txt`, `pyproject.toml`, `README.md`, `src/`, `backend/`,
  `configs/`, `models/`), plus `Dockerfile`, `.dockerignore`, `.gitattributes`, `deploy-backend.yml` and
  `scripts/deploy_scope.py`. A test fails if a Dockerfile `COPY` source is missing from the list.
- Baseline: the newest `huggingface-space` GitHub deployment that the workflow wrote. It's written only after the Space
  is `RUNNING` and `/healthz` reports `model_loaded: true`, and it names both the tested commit and the Space commit
  pushed. A run skips only if that record has `success`, the Space's head is still that commit, and no image input
  differs. Anything unknown deploys, and manual runs always deploy. The workflow adds `deployments: write` and stops
  persisting the checkout token.
- `README.md` redeploys: it's package metadata in the image and, with the config header, the Space's card (ADR-027).
- Acceptance clarifications:
  - The comparison is a tree diff, so a change reverted before it was pushed doesn't deploy.
  - The review found that comparing against the last successful deploy alone isn't enough. A deploy that pushed and
    then failed leaves the Space on its commit, and a later revert would compare as unchanged. The Space-head check
    covers that case.
- Verification:
  - Locally:
    - 67 deploy-scope tests and 98 with the backend and Makefile tests;
    - ruff and mypy clean, all three workflows parse, pre-commit passed on every commit;
    - the six regressions the tests must catch (dropping `--no-renames`, comparing with the parent, dropping the
      Space-head check, dropping `README.md`, recording with `always()`, persisting the token) each fail them;
    - real ranges: `e8c3eda..05859ca` (docs) skips, `05859ca..f01b2e0` (README) deploys, and
      `4e6a957~1..05859ca` deploys, though `05859ca` alone against its parent would skip.
  - `/code-review`: 10 findings.
    - Five fixed: the failed-deploy revert (now the Space-head check), a manual run waiting on the diff, a diff failure
      crashing the step, the lookup paging through records, and a missing step output skipping silently.
    - Token scope narrowed with `persist-credentials: false`.
    - Declined: sharing the manual-run CI check's API client (out of scope), and checking the live Space's health on a
      skip (monitoring).
  - `/security-review`: no findings.
  - Image change on `main`:
    - CI run `37819440661` on `b21b97a`: all 6 jobs green, 310 passed on 3.10 and 3.11.
    - Deploy run `37819590272` found no record and deployed: Space `RUNNING`, "Healthy: model_version=v2.0.0".
    - It recorded deployment `6942786830` (`b21b97a`, Space commit `7d45a58`, `success`).
  - Docs-only push on `main`:
    - CI run `37820623531` on `fbf55d8`: all 6 jobs green.
    - Deploy run `37820817700` skipped in 11 s, with the notice "No image input changed since b21b97a…, and the Space
      is still on it". Nothing was pushed: the Space head stayed `7d45a58`, and no new record was written.
- Not done, by scope:
  - checking the live Space's stage on a skip;
  - rebuilding for a new base image without a manual run;
  - the post-deploy caption smoke test (TASK-023).

### TASK-023 — Real-model post-deploy smoke test            [status: done] (2026-10-08)
Area: deployment · inference-api
Goal: a deploy passes only once the live Space has captioned a real image with the real model and allows the Vercel
origin through CORS.
Scope:
- After the existing health gate, send one real `POST /v1/captions` request to the live Space and check the response.
- Verify CORS from the Vercel origin: a request with `Origin: https://image-captioning-system.vercel.app` gets that
  origin back in `Access-Control-Allow-Origin`.
- Account for cold starts: the Space may be asleep or still warming up, so the check waits and retries within a
  bounded timeout before it reports a failure.
- Build on the deploy workflow as TASK-022 leaves it. Document the check in `docs/CI.md` and runbook § 8.
Acceptance criteria:
- GIVEN a deploy that passed the health gate THEN one real caption request returns HTTP 200 with a
  `CaptionResponse`-shaped body, a non-empty caption, and the same `model_version` that `/healthz` reports.
- GIVEN the Vercel origin THEN the response allows it. GIVEN an origin outside the allow-list THEN the response
  doesn't allow it.
- A sleeping or waking Space doesn't fail the deploy before the bounded timeout. An error response from a running
  Space fails it, with the status and body logged and no image bytes or tokens.
- The check needs no new secret, because the API is public.
Verification: the YAML parse check; a deploy run on `main` with the smoke step passing; evidence that the step fails
when it should (for example against a wrong expected origin), reverted before commit.
Depends on: TASK-022.
Owns: the smoke step in `.github/workflows/deploy-backend.yml`, any committed test image, `docs/CI.md`, the runbook.
Out of scope: asserting caption text or quality; load or latency testing; scheduled uptime monitoring; frontend
changes; API contract changes.
Outcome:
- `scripts/smoke_caption.py` runs as a deploy step after the health gate and before the deploy record. It runs on the
  domain the gate verified: the `health` step's `space_url` output, checked against a hostname pattern.
  - It sends one `POST /v1/captions` with a 64×64 RGB gradient PNG built in code (about 8 KB, no committed binary), the
    Vercel `Origin` and its own `x-request-id`.
  - It passes on HTTP 200 with a `CaptionResponse`-shaped body:
    - a non-empty caption, with no exact text asserted;
    - the `model_version` that `/healthz` reports;
    - a non-empty `decode_strategy` and a positive `latency_ms`;
    - the request id echoed in the header and the body;
    - the Vercel origin in `Access-Control-Allow-Origin`.
  - Retried for up to 5 minutes: connection, TLS and timeout errors, a body cut off mid-read, 502/503/504, and a
    `/healthz` reporting the model not loaded yet. No attempt runs past that deadline.
  - Anything else fails at once, with the status and the error's `detail`. A failure fails the deploy, so no baseline
    is recorded (ADR-027).
  - No token is used, and nothing of the image or caption is logged. The script joins the image inputs, as part of
    the deploy procedure. ADR-028.
- Acceptance clarifications:
  - **CORS, disallowed origin: not assertable in production.** The HF Spaces proxy answers CORS itself and reflects
    any `Origin`:
    - `https://not-allowed.invalid` got `Access-Control-Allow-Origin: https://not-allowed.invalid`;
    - its preflight got a 200 echoing the requested method.
    - The app's own `CORSMiddleware` gives that origin no header and answers its preflight with a 400.
    - The smoke test therefore checks only the allowed case. The negative case is covered against the real app in
      `test_smoke_caption.py`.
  - **Finding, left open:** the Space's `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` doesn't reach the app. `load_config`
    builds `AppConfig(**yaml)`, and pydantic-settings ranks constructor arguments above environment variables, so the
    served allow-list is `base.yaml`'s localhost origins. The SPA works today only through the proxy's reflection. The
    fix belongs in `src/captioning/config`, where it changes every `CAPTIONING__*` override, so it isn't part of this
    task. Recorded in `SECURITY.md` § Known gaps.
  - **Request id:** with no `x-request-id`, the HF proxy supplies its own (for example `OCFXAO`), so the check sends
    one and requires it echoed.
  - **Cold starts:** the step runs right after the health gate, so the Space is awake. The retry window covers a
    proxy 502/503/504 or a restart in between.
- Tests: `tests/unit/test_smoke_caption.py`, 42 tests.
  - The image is a deterministic RGB PNG, and decodes through the serving decoder (`bytes_to_tensor`) to 299×299×3.
  - The contract: a passing reply, and 17 ways to break it (HTTP 500/422, non-JSON, empty or missing caption, wrong
    model version, latency, request id, origin). A repeated origin or request-id header fails too.
  - Retries:
    - 503, then a reset, then 502, then success;
    - a cut-off body;
    - a model still loading after a restart.
  - Failures:
    - a Space that never wakes, or a model that never loads, fails at the deadline;
    - no attempt runs past the deadline;
    - a 500 fails without retrying.
  - A server message can't inject a workflow command into the log.
  - The whole check run against `create_app()` (the real CORS, request-id and body-cap middleware) with a stand-in
    predictor: it passes and receives the PNG byte for byte, and fails when the app doesn't allow the origin.
  - CLI input validation, and the workflow's step order and wiring. `test_deploy_scope.py` lists the script as an image
    input.
- Verification:
  - Locally:
    - 110 smoke and deploy-scope tests; 141 with the Makefile and backend route tests;
    - ruff and mypy clean, the workflows parse, pre-commit passed;
    - the eleven regressions the tests must catch each fail them: accepting an empty caption, dropping the CORS check,
      dropping the request-id check, retrying a 500, `continue-on-error`, the smoke step after the record, no log
      escaping, keeping only the last of a repeated header, no per-attempt deadline, not retrying a cut-off body, and
      not waiting for a loading model.
  - `/code-review`: 10 findings.
    - Fixed:
      - log escaping against workflow-command injection;
      - repeated headers;
      - the deadline bounding every attempt;
      - retrying cut-off bodies and TLS errors;
      - waiting out a model still loading;
      - one host pattern shared by the workflow and the script;
      - the summary's CORS wording.
    - Declined:
      - a repository variable for the origin: new configuration, and a test keeps the two copies equal;
      - reusing the health step's model version: the script stays usable by hand;
      - moving the decoder test out of this file.
  - Live, before pushing, the script against the production Space, twice:
    - first run, before the review fixes: 1170 ms inference;
    - final code: "Captioned a 8031-byte PNG in 3.2s: HTTP 200, 9-word caption from model v2.0.0 (greedy, 980 ms
      inference), x-request-id echoed, Access-Control-Allow-Origin matched https://image-captioning-system.vercel.app."
- Not done, by scope:
  - asserting caption text or quality;
  - latency budgets;
  - uptime monitoring between deploys;
  - fixing the CORS env-override precedence;
  - the README's 4E line, left for the owner.
