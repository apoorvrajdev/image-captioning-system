# Phase 0 — Bootstrap (decision log)

> Phase 0 establishes the engineering scaffolding the rest of the project will
> stand on. Nothing here changes the model; everything here changes how the
> repo *looks and behaves* to the next person who clones it (including
> recruiters and CI runners).

## What this phase delivers

| Artefact | Purpose |
|---|---|
| [`notebooks/01_ieee_inceptionv3_transformer.ipynb`](../notebooks/01_ieee_inceptionv3_transformer.ipynb) | Renamed from `image-captionin-using-dl.ipynb` via `git mv` to preserve history. Now the canonical, frozen IEEE artefact. |
| [`notebooks/README.md`](../notebooks/README.md) | Documents the frozen-notebook policy and conventions for any new notebooks. |
| [`pyproject.toml`](../pyproject.toml) | Single source of truth for the `captioning` Python package, dependency groups, and tool config (ruff/mypy/pytest/coverage). |
| [`requirements.txt`](../requirements.txt) | Pinned runtime deps, used directly by Docker and CI (mirrors `[project.dependencies]`). |
| [`requirements-dev.txt`](../requirements-dev.txt) | Pinned dev deps (lint, type-check, test, hooks). |
| [`requirements-eval.txt`](../requirements-eval.txt) | Pinned metric deps, kept separate to avoid bloating the serving image. |
| [`.python-version`](../.python-version) | Pins Python 3.10 for `pyenv` users. |
| [`.env.example`](../.env.example) | Schema for `pydantic-settings`-loaded env vars. |
| [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) | Hooks: ruff, mypy, nbstripout, prettier (frontend), gitleaks. |
| [`Makefile`](../Makefile) | Discoverable command index (`make help`). |
| [`LICENSE`](../LICENSE) | MIT license, attribution to original author. |
| [`.gitignore`](../.gitignore) | Production-grade exclusions, organised by purpose with explanatory comments. |
| [`docs/restructure-plan.md`](./restructure-plan.md) | Public-facing engineering plan for Phases 0–4. |

---

## Decisions and reasoning

### 1. Why `src/` layout over flat layout?

A flat layout (`captioning/` at repo root) lets test code accidentally import
from the working tree instead of the *installed* package. That hides bugs that
would only surface in production, where the tree layout is gone. The `src/`
layout forces every test, every script, and every import to go through the
installed package — exactly the path users will follow. This is the layout
the [Python Packaging Authority recommends](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/),
and it's what production Python codebases (FastAPI, Pydantic, HTTPX) use.

### 2. Why `pyproject.toml` AND `requirements.txt`?

They serve different audiences:

- **`pyproject.toml`** is the *source of truth* for the package — its name,
  version, abstract dependency ranges, optional extras, and tool configuration.
  When you `pip install -e .[dev]`, this is what pip reads.
- **`requirements.txt`** is the *concretely pinned snapshot* — used by Docker
  builds, CI runners, and anyone who wants `pip install -r requirements.txt`
  without cloning the source. It's regenerable from `pyproject.toml` via
  `pip-compile`, but committing it explicitly makes installs deterministic and
  diffable.

Phase 5+ will switch to `pip-compile` for automated regeneration; for now,
manual mirroring is simpler and beginner-readable.

### 3. Why pin `tensorflow-cpu==2.15.0` so hard?

Two independent reasons stack:

1. **`tensorflow-cpu` (not `tensorflow`)**: the GPU build pulls ~600 MB of
   CUDA libraries that are useless on CPU-only HuggingFace Spaces. Splitting
   the wheel keeps the serving image well under 1.5 GB.
2. **2.15 specifically**: TF 2.16 swapped to Keras 3 by default. The IEEE
   notebook uses `tf.keras.layers.TextVectorization` with the Keras 2
   save/load API. Upgrading silently changes vocab serialisation, which
   silently changes BLEU. Pinning is the difference between
   *reproducible-published-result* and *reproducibility theatre*.

When Phase 5+ migrates to a modern multimodal backbone, this pin will move
in a deliberate, tested step — not by accident.

### 4. Why Ruff over Black + isort + flake8?

Ruff replaces all three with one tool that runs ~100x faster, reads config
from a single section in `pyproject.toml`, and ships its own formatter
(`ruff format`) that is byte-identical to Black's output. One install, one
config, one cache. Recruiters reading the repo see the modern Python tool;
CI runs faster; `make format` is one command, not three.

### 5. Why `nbstripout` is non-negotiable in pre-commit

Notebook outputs include base64-encoded images, full DataFrames, and
sometimes credentials printed by accident. Committed notebook diffs without
output stripping are unreadable (`+aaaaaaaaaa[base64]+aaaaa…`) and
occasionally leak data. `nbstripout` removes all output cells on commit,
keeping notebook history clean and reviewable.

### 6. Why include a `Makefile` on a Windows project?

Three reasons:

1. **CI runs on Linux** — every CI job uses the same Make targets, so the
   commands you run locally match what CI runs.
2. **Discoverability** — `make help` is one command that prints every
   high-level operation with a one-line description. A new contributor (or
   recruiter cloning the repo) sees the entire workflow in one screen.
3. **Tooling availability** — Make is a 5-second install on Windows
   (`winget install GnuWin32.Make`, Git Bash, or WSL). PowerShell users who
   skip Make can still read the Makefile and run the underlying commands
   directly.

### 7. Why a `freeze-paper-notebook` Make target?

The IEEE paper points reviewers at the notebook. If the notebook drifts from
what the paper describes, reviewers running it will see numbers that don't
match the paper — and that's a scientific integrity issue, not a software
issue. The target hashes the notebook and asserts it matches a locked
SHA-256. Phase 4 wires this into CI as a required check on `main`.

### 8. Why split optional deps into `[hf]`, `[eval]`, `[mlflow]`, `[dev]`?

The slim production image (`backend:latest`) does NOT need transformers,
torch, pycocoevalcap, or MLflow. Bundling them adds ~1.5 GB of dependencies
the production code never imports. Extras let `pip install -e ".[hf]"` add
the HuggingFace baselines for the Phase 3 comparison demo, while
`pip install -r requirements.txt` keeps the production install lean.

### 9. Why MIT license?

The IEEE paper is published under IEEE's standard terms; the *code* is
covered separately. MIT is the most permissive widely recognised license —
it lets recruiters, students, and other researchers freely fork, learn from,
and extend the code. For a recruiter-grade portfolio project, permissive
licensing signals "I want this work to be useful," which is the right tone.

### 10. Why folder name `configs/` (plural), not `config/` (singular)?

`config/` was the empty folder shipped with the template. The plural form
`configs/` is the convention in modern Python ML projects (FastAPI's own
example apps, Hydra projects, the official `transformers` repo) because
it holds multiple files (one per environment, model variant, or run).
Phase 1 creates `configs/` with content; the empty `config/` folder will
be removed in the Phase 1 commit that introduces the YAML files.

---

## What this phase deliberately does NOT do

- **No code is moved out of the notebook yet.** That's Phase 1, behind a
  parity validation gate.
- **No `src/captioning/` modules are created.** Empty `__init__.py` files
  would just be churn; Phase 1 will create them with real code.
- **No Dockerfile or docker-compose.yml.** They depend on `backend/app/`
  existing; both arrive in Phase 1.
- **No GitHub Actions workflows.** They live in Phase 2, after there is
  Python code to lint and type-check.
- **No README rewrite.** The current README accurately describes the
  research; the demo-link rewrite happens in Phase 2 once a live URL exists.

This restraint is deliberate. Each phase ships a coherent slice of value;
running ahead would create half-built features and vague commits.

---

## Local setup checklist for the developer

After pulling this commit, on a fresh dev box:

```bash
# 1. Create a Python 3.10 virtual environment.
python -m venv .venv
.venv\Scripts\activate              # PowerShell
# source .venv/bin/activate         # Linux/macOS

# 2. Install dev dependencies + the package (editable).
make install-dev
# Or, without Make:
#   pip install -r requirements-dev.txt -r requirements-eval.txt
#   pip install -e ".[hf,mlflow]"

# 3. Register pre-commit hooks.
make install-hooks
# Or:  pre-commit install

# 4. (Optional) Lock the paper notebook's hash, so CI can enforce parity.
make lock-paper-notebook

# 5. Verify everything works.
make pre-commit                     # Run all hooks against all files
make test                           # No tests yet — exits cleanly with "no tests collected"
```

The first `make install-dev` will take a few minutes (TensorFlow is large).
Subsequent runs hit the wheel cache and complete in seconds.
