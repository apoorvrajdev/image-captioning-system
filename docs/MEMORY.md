# Project memory — current state

> Living document: **current state only**. Permanent decisions → [`DECISIONS.md`](DECISIONS.md).
> Backlog → [`TASKS.md`](TASKS.md). Update at the end of every task.

_Last updated: 2026-10-07_

## Current phase

- **Completed:** Phase 0 (bootstrap), Phase 1 (modularisation), Phase 1b (training stabilisation +
  metric suite + stabilized checkpoint), Phase 2A (FastAPI), Phase 2B (SPA), Phase 2C (public deployment),
  Stage 0 evaluation-methodology gate (verdict: **reframe, do not retrain**), engineering-workflow setup.
- **Next:** Phase 3 — multimodal baselines (3A–3D), decomposed into TASK-009 – TASK-018 plus TASK-007 in
  [`TASKS.md`](TASKS.md). The evaluation protocol (TASK-009) is recorded in `EVAL_METHODOLOGY.md` § 8 and ADR-019.
  The slice loader (TASK-010), captioner adapters (TASK-011), comparison runner (TASK-012) and cross-run summary
  (TASK-013) exist. The first baseline results are committed (TASK-014): `results/phase3-comparison/`, recorded in
  `EVAL_METHODOLOGY.md` § 8.8. The latency benchmark tooling exists and is tested (TASK-015), with its protocol in
  `EVAL_METHODOLOGY.md` § 9 and ADR-020. TASK-016 is done:
  - The four CPU latency runs (`results/phase3-latency-*-greedy-cpu/`, § 9.9) and the four Kaggle T4 GPU runs
    (`results/phase3-latency-*-greedy-cuda/`, § 9.10) are committed.
  - They come from different hosts and runtimes, so they aren't a controlled CPU-vs-GPU comparison.
  - The CNN's batch-8 figures are sequential single-image calls.
  - TASK-017 is done: `python -m scripts.export_dashboard_data` writes the SPA's static
    `frontend/src/generated/phase3-dashboard.json` from those results (ADR-021), and a test fails if it drifts.
  - TASK-018 (dashboard UI) is next and hasn't started. TASK-007 (Playwright) still awaits install approval.
- **Current task:** none in progress. TASK-008 (backend deploy) is done: deployed and verified 2026-10-03.
  TASK-004 (model-version labelling) is done: `v2.0.0` verified live. TASK-006 (bounded upload read) is done and
  deployed. TASK-005 (Makefile repair) is done. TASK-007 (Playwright) is deferred to the start of Phase 3D.

## System status (verified 2026-10-03, local Windows, Python 3.10.11)

| Check | Result |
|---|---|
| `pytest tests backend/app/tests` | 238 passed on 2026-10-07, after TASK-017 (1 pydantic `model_` namespace warning) |
| ruff lint + format check | clean (103 files, 2026-10-07) |
| mypy (pyproject config, `strict = false`) | 0 errors, 83 files (2026-10-07) |
| Parity audit (`scripts/notebook_module_audit.py`) | 4/4 |
| Notebook SHA-256 freeze | OK |
| `SKIP=mypy pre-commit run --all-files` (clean clone, LF) | all hooks pass |
| Frontend `npm run lint` / `npm run build` | clean / builds |
| CI on `main` for `915112b` (run `37140358717`) | green, all 6 jobs incl. `pre-commit` |
| `deploy-backend.yml` (run `37140993110`, manual, `915112b`) | **success**: Space commit `123c5aa`, health gate passed |
| Backend Space (HF runtime API, 2026-10-03) | **`RUNNING`** (cpu-basic), no error message |
| `GET /healthz` (public, 2026-10-03T18:12Z) | HTTP 200, `model_loaded: true`, `model_version: v2.0.0` |

SPA on Vercel. The API's HF Space (Docker, cpu-basic) is **live** at
`https://apoorvrajdev-image-captioning-api.hf.space` (`/healthz`, `/docs`, `/openapi.json` all HTTP 200). `deploy-backend.yml`
is enabled and set to auto-deploy every CI-green commit on `main` (ADR-017). Both paths have run successfully: manual (run `37140993110`) and automatic (run `37144272026`), each passing the live health gate. The Space serves HF Hub
`apoorvrajdev/captioning-inceptionv3-transformer` at tag `v2.0.0` (commit `59d93b4`) and reports `model_version: v2.0.0`
(TASK-004). Headline results: `results/stabilized-greedy/`, `results/stabilized-beam-w4-lp07-rp12/`
(beam CIDEr 0.826; 5-ref BLEU-4 25.91). The Phase 3 baseline comparison is `results/phase3-comparison/`. It is not a
held-out comparison (`EVAL_METHODOLOGY.md` § 8.5).

## Recent changes

- 2026-10-07 Phase 3 dashboard data (TASK-017, done):
  - `captioning.evaluation.dashboard` and `python -m scripts.export_dashboard_data` turn `results/phase3-comparison/`
    and the eight `results/phase3-latency-*/` runs into `frontend/src/generated/phase3-dashboard.json`, which the SPA
    will import at build time. There is no backend endpoint (ADR-021).
  - Per model: display name, Hub id and revision, metrics, latency per device and batch size, and source run ids.
    Shared: the slice description, the § 8.5 overlap caveat and the §§ 8–9 notes. Values are copied verbatim.
  - Inconsistent sources are refused. `tests/unit/test_dashboard_export.py` fails if the committed file drifts from
    `results/`, so a new comparison summary or latency run needs a re-export in the same change.
  - The file is in `generated/` because `.gitignore`'s `data/` rule matches `frontend/src/data/`. Prettier's
    pre-commit hook skips that directory.
  - Committed as `2137624`..`313ae79`, plus the closing docs.
- 2026-10-07 Phase 3 GPU latency runs (TASK-016, GPU half; TASK-016 is done):
  - The same four models, pinned revisions, 32 images and § 9 protocol as the CPU half, with `--device cuda`.
  - Run in a dedicated private Kaggle kernel, `apoorvujjwal/task-016-phase-3-gpu-latency-benchmark`, on one Tesla T4
    (GPU 0 of 2).
  - Runtime: the Kaggle image is Python 3.13, so the kernel built two uv Python 3.10.20 environments.
    - Hugging Face models: `torch` 2.3.0+cu121.
    - CNN: `tensorflow` 2.15.0 with its `and-cuda` CUDA pins, without TensorRT.
    - Two environments were needed because the two frameworks pin conflicting CUDA libraries.
  - Checks before the runs: `torch` CUDA and the TensorFlow GPU both ran real operations; the revisions and the
    checkpoint SHA-256 matched.
  - Output: `results/phase3-latency-*-greedy-cuda/`, with statistics in `EVAL_METHODOLOGY.md` § 9.10.
  - The CNN's GPU load time includes Keras downloading the ImageNet InceptionV3 weights on the fresh machine.
  - Committed as `4ba72c8`..`04a190f`, plus the closing docs.
- 2026-10-07 Phase 3 CPU latency runs (TASK-016, CPU half):
  - Real runs of BLIP-base, ViT-GPT2, GIT-base-coco and the CNN + Transformer at their pinned revisions, under the
    unchanged § 9 protocol: one `scripts/benchmark_latency.py --device cpu` invocation each.
  - Host: the owner's laptop, an AMD Ryzen 7 7435HS on Windows 11, CPU only. Weights came from the local cache with
    `HF_HUB_OFFLINE=1`.
  - Output: `results/phase3-latency-{blip-base,vit-gpt2,git-base-coco,inceptionv3-transformer-stabilized}-greedy-cpu/`,
    160 batch-1 and 20 batch-8 samples each, with the statistics in `EVAL_METHODOLOGY.md` § 9.9.
  - The CNN's batch-8 figures are sequential single-image calls (§ 9.5). No ranking or cross-device claim is made.
  - Committed as `672b224`..`11e749c`.
- 2026-10-07 latency benchmark tooling (TASK-015, done; no measurements):
  - `scripts/benchmark_latency.py` times one model per invocation through the shared `Captioner.caption()` call, using
    `captioning.evaluation.latency`.
  - It writes only `results/phase3-latency-<model_id>-<decoding>-<device>/latency.json`. Quality runs are untouched.
  - Protocol (`EVAL_METHODOLOGY.md` § 9, ADR-020):
    - the first 32 slice images, batch sizes 1 and 8;
    - 1 untimed warmup pass, then 5 measured passes;
    - `time.perf_counter`, with load time recorded separately;
    - count, mean, median, min and max, plus the raw samples, with nothing filtered.
  - The CNN + Transformer captions a batch one image at a time, so its batch figures aren't like-for-like with the
    Hugging Face models' (§ 9.5).
  - Committed as `b191f4a`..`dc831da`.
- 2026-10-06 Phase 3 baseline runs (TASK-014, done):
  - Real greedy runs, under the unchanged § 8 protocol, of BLIP-base, ViT-GPT2 and GIT-base-coco at their pinned
    revisions: `results/phase3-{blip-base,vit-gpt2,git-base-coco}-greedy/`.
  - `results/phase3-inceptionv3-transformer-stabilized-greedy/` re-runs the CNN + Transformer (Hub `v2.0.0`) through
    the harness. It reproduces `results/stabilized-greedy/` exactly: 500/500 predictions, with bit-identical metrics.
  - `results/phase3-comparison/` passes the slice check (500 images, 732 references, `6b5628bf…`) and states the
    overlap caveat. `EVAL_METHODOLOGY.md` § 8.8 has the details.
  - Run on a local CPU, with the slice images fetched from the same Kaggle COCO 2017 dataset.
  - Committed as `a1ab352`..`ba888b3`.
- 2026-10-06 cross-run summary (TASK-013, done):
  - `scripts/compare_runs.py` joins Phase 3 run directories into `comparison.json` and `comparison.md`.
  - It refuses any run whose slice fingerprint, counts, protocol or normalisation differ, or whose files are missing or
    malformed.
  - The committed CNN + Transformer runs enter only as `--reference-run` rows.
  - Committed as `7885850`..`7d2b228`.
- 2026-10-06 comparison runner (TASK-012, done):
  - `scripts/compare_models.py` captions the committed slice with selected `config.compare` models.
  - It writes one new `results/phase3-<model_id>-<decoding>/` per model: the five standard files plus
    `comparison_meta.json` (slice fingerprint, revision, decode settings).
  - It checks slice counts, model ids, run-directory collisions and image presence before any model loads.
  - Committed as `c7d71f9` and `445c183`.
- 2026-10-05 captioner adapters (TASK-011, done):
  - `captioning.baselines` gives every compared model one `Captioner` interface: `CNNCaptioner` wraps
    `CaptionPredictor` unchanged, and `HFCaptioner` loads a pinned Hub revision with the protocol's decode
    settings.
  - Captions are normalised once via `preprocess_caption` → `strip_sentinels`.
  - `torch`/`transformers` are imported only in `HFCaptioner.load()`.
  - The protocol values live in a strict `compare` config section (`configs/base.yaml`).
  - Committed as `8f389e0`..`ebe4ea3`.
- 2026-10-05 evaluation-slice loader (TASK-010, done):
  - `captioning.evaluation.load_eval_slice` reads the Phase 3 slice from a committed `predictions.jsonl`, keeping
    file order and stored references and remapping images by file name.
  - A test pins the shared greedy/beam slice: 500 images, 732 references.
  - Committed as `c6f9d7e` and `99f3dfb`.
- 2026-10-04 Makefile repair (TASK-005, done):
  - `docker-build` builds the root `Dockerfile`.
  - `docker-build-hf`, `docker-up` and `docker-down` are removed: there was no `ARG INSTALL_HF` and no compose file.
  - `eval` and `predict` pass `--config`, `--weights` and `--tokenizer-dir`, using `MODEL_DIR ?= models/v1.0.0`.
  - `tests/unit/test_makefile.py` checks the targets statically.
  - Committed as `1be8c1f`..`b7b2903`.
- 2026-10-04 bounded upload read (TASK-006, done): `/v1/captions` reads at most `max_upload_bytes + 1` bytes, then
  returns a 413 if the upload is over the limit. The 413 detail now reads "Image exceeds the {limit}-byte upload
  limit."; status codes and the response shape are unchanged. Committed as `cd5ee1c` (fix) and `b42fae6` (two new
  tests). Starlette still spools the whole multipart body before the route runs (see [`SECURITY.md`](SECURITY.md)).
- 2026-10-03 model-version labelling (TASK-004, done): the Space variable `BACKEND_MODEL_VERSION=v2.0.0` now matches
  `BACKEND_WEIGHTS_HUB_REVISION=v2.0.0`. `/healthz` reports `model_version: v2.0.0` with `model_loaded: true`. The
  runbook's promotion and rollback steps now move both variables together (ADR-018). No code change, no redeploy.
- 2026-10-03 deploy fix (TASK-008, done): `deploy-backend.yml` deploys the tested SHA (manual runs must
  prove the exact SHA passed CI), skips superseded commits, adds the Space config header to a force-pushed deploy
  commit, and gates on HF `RUNNING` + `/healthz` `model_loaded: true` (ADR-017).
  - Committed as `a2ee431`..`915112b`; CI run `37140358717` green.
  - The first deploy (run `37140993110`) created Space commit `123c5aa` and resolved the `CONFIG_ERROR`.
- 2026-10-03 workflow upgrade (committed and pushed as `545c76b`..`9e8874c`):
  - `.claude/` config is now tracked (ADR-015), and `.claude/settings.json` enforces the invariants.
  - The code index rebuilds at session start.
  - CI gained a `pre-commit` job (ADR-016).
  - `CLAUDE.md` gained a debugging protocol and a review step.
  - `ship-task` gained per-change-type flows.
  - README drift fixed (TASK-003).
- 2026-09-24: living docs added, parity audit wired into CI, frozen notebook pinned to LF in `.gitattributes`.
- Stage 0 eval audit landed (`docs/EVAL_METHODOLOGY.md`, `results/stabilized-beam-w4-lp07-rp12/verdict.md`).

## Known issues / open debt

- No frontend tests / e2e (TASK-007), no coverage measured in CI, no dependency-vulnerability scanning,
  and no full-history secret scan in CI.
- Pydantic warning: `BackendSettings.model_version` (`backend/app/core/config.py`) collides with the protected
  `model_` namespace (harmless; the response schemas already set `protected_namespaces=()`).
- `README.md` doesn't cite the Phase 3 results yet. TASK-014 left it untouched because it holds owner-staged edits.

## Before coding, know this

- Windows + Git Bash. `make` isn't on PATH, but `mingw32-make -n <target>` (MSYS2) dry-runs a target. Use
  `.venv/Scripts/*.exe` (see `CLAUDE.md` → Commands).
- The notebook is frozen, parity must stay 4/4, and `results/` + `models/vX.Y.Z/` are immutable. Edits to
  them are blocked by `.claude/settings.json`.
- Retraining and deployments are owner-run (Kaggle / HF / Vercel). Code tasks prepare instructions only.
- Promoting or rolling back weights means setting the Space's `BACKEND_WEIGHTS_HUB_REVISION` and `BACKEND_MODEL_VERSION`
  to the same tag (runbook § 3). The code never links them.
