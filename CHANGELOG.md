# Changelog

All notable changes to `nexus-context` will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] — 2026-08-18

### Changed
- `README.md`: Comprehensive rewrite for PyPI — covers problem statement with analogies,
  full quick-start, all 7 features explained, CLI reference, API reference, benchmark table,
  architecture deep-dive, backend compatibility table, contributing guide, and changelog link.

---



---

## [0.2.0] — 2026-08-18

### Added
- **Real-Time Observability Dashboard** (`/dashboard`) — dark professional UI with live SSE metric panels:
  pipeline latency sparkline, token budget gauge, KV cache hit rate, memory pool counter,
  graph topology counters, tool interception log, chunk boundary event feed, LTKB fact feed.
- **Tool Call Interception & Compression** — `ToolCallCompressor` intercepts `role="tool"` messages
  exceeding 512 tokens before budget calculation. JSON array truncation (max 3 items) and
  sentence-boundary text fallback. Reduces tool output tokens by 60–80%.
- **Session Persistence & Crash Recovery** — `SessionStore` (async SQLite via `asyncio.to_thread`)
  persists session state every N turns. Full session restore on `nexus-serve` restart.
  New CLI flags: `--persist`, `--db-path`, `--persist-every`.
- **Multi-Modal Context Graph (Feature A)** — New node types `TOOL_JSON_FIELD`, `API_RESPONSE`
  and edge types `TOOL_FIELD_TO_CODE_REF`, `SCHEMA_TO_TOOL_RETURN`. Cross-modal JSON field →
  AST assignment edges via `extract_tool_return_nodes()`.
- **LLM-Agnostic Token Budget Estimation** — `TokenizerRegistry` with `@lru_cache` for model-specific
  HuggingFace tokenizer loading. Automatic model family detection (Qwen, LLaMA, Mistral, Gemma, Phi).
  Falls back to tiktoken `cl100k_base`, then character-estimate heuristic.
- **Cross-Session Long-Term Knowledge Base** — `LongTermKnowledgeBase` (SQLite) extracts high-weight
  graph nodes (retention_weight ≥ 0.8, in-degree ≥ 2) at session end. TF-IDF retrieval of relevant
  facts injected into Zone P of new sessions. New CLI flags: `--ltkb-db`, `--no-ltkb`.
- **Async Telemetry Event Bus** — `TelemetryBus` (asyncio.Queue fan-out) with `get_recent_history()`,
  subscriber management, history replay on SSE connect, and emit helper functions.
- **Public `create_app()` factory** — Top-level `nexus_context.create_app()` for embedding the
  middleware in existing FastAPI applications.
- 39 new unit tests (115 total). All passing.

### Changed
- `pyproject.toml`: Moved heavy ML dependencies (`spacy`, `sentence-transformers`, `torch`,
  `transformers`, `tree-sitter-*`) to optional extras. Core install is now ~50MB instead of ~5GB.
- `__init__.py`: Updated `__version__` to `0.2.0`, added `__email__`, `create_app` export.
- `middleware.py`: Integrated all 6 feature hooks into the 5-stage request pipeline.

### Fixed
- `BlockAligner` correctly handles edge case where Zone P is already block-aligned (no-op padding).
- `SubmodularSolver` TF-IDF fallback activates gracefully when `sentence-transformers` not installed.

---

## [0.1.0] — 2026-08-01

### Added
- Initial alpha release.
- `nexus.guard`: AST dependency graph builder (`ContextGraphBuilder`) for Python, SQL, Bash, JS.
  Submodular compaction solver (`SubmodularSolver`) with γ→∞ dangling-reference penalty.
- `nexus.cache`: Block-aligned Zone P prefix freezing (`BlockAligner`). Zone P/T/R segmentation
  (`ZoneSegmenter`). OpenAI-compatible `/v1/chat/completions` proxy endpoint.
- `nexus.memory`: WWW episodic-to-semantic decay (`MemoryPool`, `WWWParser`).
- `nexus-serve` CLI entry-point (Uvicorn + FastAPI).
- Support for vLLM, SGLang, and Ollama backends via `--backend-type` flag.
- 76 unit tests.
