# =============================================================================
# Makefile — common project commands.
# -----------------------------------------------------------------------------
# Why a Makefile when the team uses Windows + PowerShell?
#   1. CI (Linux) runs these targets directly.
#   2. The file is the canonical, discoverable command index — `make help`
#      tells a new contributor (or a recruiter cloning the repo) the entire
#      development workflow in one screen.
#   3. Windows users can install Make via `winget install GnuWin32.Make`,
#      use Git Bash, WSL, or just read the `RUN:` lines and run the underlying
#      command in PowerShell directly.
#
# Conventions:
#   - `.PHONY` declares targets that don't produce a same-named file.
#   - Target naming: `verb-noun` (e.g. `docker-build`, not `build_docker`).
#   - Each target is annotated with a one-line `## description` comment that
#     `make help` parses and prints automatically.
# =============================================================================

# Default Python interpreter. Override on Windows: `make PYTHON=py install`.
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
NPM    ?= npm

# Directories
SRC_DIR      := src/captioning
BACKEND_DIR  := backend
FRONTEND_DIR := frontend
TESTS_DIR    := tests
NOTEBOOK_FROZEN := notebooks/01_ieee_inceptionv3_transformer.ipynb

# ---- Default goal: show available targets -----------------------------------
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Image Captioning System — available commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Install / setup
# =============================================================================

.PHONY: install
install: ## Install runtime dependencies only (slim, for Docker parity)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install runtime + dev + eval extras + the captioning package (editable)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt -r requirements-eval.txt
	$(PIP) install -e ".[hf,mlflow]"

.PHONY: install-hooks
install-hooks: ## Register pre-commit hooks in .git/hooks/
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# Code quality
# =============================================================================

.PHONY: lint
lint: ## Run ruff lint checks (no fixes)
	ruff check $(SRC_DIR) $(BACKEND_DIR) scripts $(TESTS_DIR)

.PHONY: format
format: ## Auto-fix lint issues and reformat
	ruff check --fix $(SRC_DIR) $(BACKEND_DIR) scripts $(TESTS_DIR)
	ruff format $(SRC_DIR) $(BACKEND_DIR) scripts $(TESTS_DIR)

.PHONY: typecheck
typecheck: ## Run mypy static type checks
	mypy $(SRC_DIR) $(BACKEND_DIR)/app scripts

.PHONY: pre-commit
pre-commit: ## Run all pre-commit hooks against ALL files
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: ## Run pytest (fast, unit + integration)
	pytest $(TESTS_DIR) $(BACKEND_DIR)/app/tests -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest $(TESTS_DIR) $(BACKEND_DIR)/app/tests \
		--cov=$(SRC_DIR) --cov=$(BACKEND_DIR)/app \
		--cov-report=term-missing --cov-report=xml --cov-report=html

.PHONY: test-smoke
test-smoke: ## Run only the fast smoke tests (used by Docker HEALTHCHECK CI step)
	pytest $(TESTS_DIR) -v -m "not slow" --maxfail=1

# =============================================================================
# ML lifecycle (Phase 1+ — placeholders until scripts/ exists)
# =============================================================================

.PHONY: train
train: ## Train the IEEE InceptionV3+Transformer model from configs/base.yaml
	$(PYTHON) -m scripts.train --config configs/base.yaml

.PHONY: eval
eval: ## Evaluate the latest model on COCO val (BLEU, CIDEr, METEOR, ROUGE)
	$(PYTHON) -m scripts.evaluate --config configs/base.yaml --report docs/results/latest.md

.PHONY: predict
predict: ## CLI single-image inference (usage: make predict IMAGE=path/to/img.jpg)
	$(PYTHON) -m scripts.predict --image $(IMAGE)

# =============================================================================
# Backend (FastAPI)
# =============================================================================

.PHONY: serve
serve: ## Run the FastAPI backend locally with hot reload
	uvicorn app.main:app --app-dir $(BACKEND_DIR) --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Docker
# =============================================================================

.PHONY: docker-build
docker-build: ## Build the backend Docker image (slim, no HF extras)
	docker build -f $(BACKEND_DIR)/Dockerfile -t captioning-backend:latest .

.PHONY: docker-build-hf
docker-build-hf: ## Build the backend image WITH HuggingFace baselines (~2.3 GB)
	docker build --build-arg INSTALL_HF=1 -f $(BACKEND_DIR)/Dockerfile -t captioning-backend:hf-latest .

.PHONY: docker-up
docker-up: ## Start backend + frontend + mlflow via docker compose
	docker compose up --build

.PHONY: docker-down
docker-down: ## Stop docker compose stack
	docker compose down

# =============================================================================
# Reproducibility / paper integrity
# =============================================================================

.PHONY: freeze-paper-notebook
freeze-paper-notebook: ## CI guard: assert the IEEE notebook hasn't been modified
	@$(PYTHON) -c "import hashlib, sys; \
h = hashlib.sha256(open('$(NOTEBOOK_FROZEN)', 'rb').read()).hexdigest(); \
expected = open('.paper-notebook.sha256').read().strip() if __import__('os').path.exists('.paper-notebook.sha256') else None; \
sys.exit(0) if expected is None else (print(f'ERROR: notebook hash {h} != frozen {expected}') or sys.exit(1)) if h != expected else (print('OK: paper notebook is byte-stable'), sys.exit(0))"

.PHONY: lock-paper-notebook
lock-paper-notebook: ## Record the current notebook hash as the frozen reference
	@$(PYTHON) -c "import hashlib; \
h = hashlib.sha256(open('$(NOTEBOOK_FROZEN)', 'rb').read()).hexdigest(); \
open('.paper-notebook.sha256', 'w').write(h + '\n'); \
print(f'Locked paper notebook at {h}')"

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Remove build artefacts, caches, and test outputs (NOT models/)
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## clean + remove mlruns/, outputs/, and downloaded models cache
	rm -rf mlruns/ outputs/ models/cache/
