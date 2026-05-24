# DeepRead — common dev tasks.
#
# All Python tooling routes through `uv` so the project's lockfile is the
# single source of truth. `make help` lists every target.

UV       ?= uv
PYTHON   ?= $(UV) run python
PYTEST   ?= $(UV) run pytest
PORT     ?= 8000

.DEFAULT_GOAL := help

# ── setup ────────────────────────────────────────────────────────────

.PHONY: install
install: ## Sync dependencies from pyproject.toml + uv.lock
	$(UV) sync

.PHONY: papers
papers: ## (Re)fetch the bundled classic papers into papers_pdf_download/
	$(PYTHON) scripts/refresh_papers.py

# ── run ──────────────────────────────────────────────────────────────

.PHONY: run
run: ## Launch Chainlit on $(PORT) (default 8000)
	$(UV) run chainlit run app.py -h --port $(PORT)

.PHONY: dev
dev: ## Launch Chainlit with hot-reload on file changes
	$(UV) run chainlit run app.py -w --port $(PORT)

# ── quality ──────────────────────────────────────────────────────────

.PHONY: test
test: ## Run the full test suite
	$(PYTEST) -q

.PHONY: smoke
smoke: ## Smoke-test text, vision, and PDF pipelines against the local Ollama
	$(PYTHON) scripts/smoke_text.py
	$(PYTHON) scripts/smoke_vision.py
	$(PYTHON) scripts/smoke_pdf.py

.PHONY: smoketest
smoketest: ## One-shot end-to-end pipeline check (ingest + budget + Ollama call)
	$(PYTHON) -m deepread.smoketest

# ── benchmarks ───────────────────────────────────────────────────────

.PHONY: bench
bench: ## Run the context-window sweep (writes benchmarks/results.json)
	$(PYTHON) benchmarks/run_context_sweep.py \
		--models gemma4:e4b --ctx 5000 20000 60000 100000 --needles 5

.PHONY: bench-quick
bench-quick: ## Fast sweep — 2 zones, 3 needles (sanity check only)
	$(PYTHON) benchmarks/run_context_sweep.py \
		--models gemma4:e4b --ctx 5000 20000 --needles 3

.PHONY: plot
plot: ## Render benchmarks/plot.png from results.json
	$(PYTHON) benchmarks/plot.py

# ── housekeeping ─────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove pytest/python caches and Chainlit runtime artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .files .chainlit/.chat_history .chainlit/.cache

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "DeepRead — Makefile targets\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
