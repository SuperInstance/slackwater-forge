# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-04

### Added
- **Core forge engine** (`forge.py`) — round-robin job execution against Ollama with time/iteration limits, continuous mode, dry-run support, graceful signal handling (SIGINT/SIGTERM), and state persistence
- **Ollama client** (`models.py`) — synchronous HTTP client supporting `/api/generate`, `/api/chat`, model listing, and model pulling; context manager support; configurable timeout
- **Job spec system** (`jobs.py`) — Pydantic-based `JobSpec` and `ForgeSession` models, `JobManager` for disk CRUD, 5 built-in templates (code review, creative writing, research, brainstorm, documentation)
- **Morning briefing synthesizer** (`briefer.py`) — artifact analysis with AI (Ollama) and offline modes, executive summary, priority items, recommendations, confidence scoring, markdown + styled HTML output
- **CLI** (`cli.py`) — Click-based with Rich terminal output; commands: run, brief, job (create/list/show/add/remove/delete), status, models, test
- **Artifact tracking** — every output saved as `.md` + `.json` metadata (model, tokens, timing, priority)
- **Signal handling** — graceful shutdown on Ctrl-C, finishing current iteration
- **Session persistence** — `.forge-state.json` tracks running state
- **Environment variable support** — OLLAMA_HOST, OLLAMA_PORT, FORGE_OUTPUT
- Example session JSON
- 10 test cases covering artifact, forge, jobs, and models
- MIT license
