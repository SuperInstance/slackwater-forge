# Contributing to Slackwater Forge

Thanks for your interest in improving Slackwater Forge!

## Getting Started

```bash
git clone https://github.com/SuperInstance/slackwater-forge.git
cd slackwater-forge
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- At least one pulled model: `ollama pull granite3.1-dense:2b`

## Development Workflow

```bash
# Run tests
pytest

# Run the forge in dry-run mode (no Ollama needed)
forge run --dry-run

# Test a single command
forge test -m granite3.1-dense:2b

# Generate a briefing from test data
forge brief --no-ai
```

### Code Style

- **Python 3.10+** with `from __future__ import annotations`
- **Type hints** on all function signatures
- **Dataclasses** for structured data (Artifact, ForgeStats, RouteResult)
- **Pydantic models** for serialized data (JobSpec, ForgeSession)
- **Click** for CLI commands
- **Rich** for terminal output
- Every public function has a **docstring**

### Architecture

```
slackwater_forge/
├── cli.py        — Click CLI entry point
├── forge.py      — Core loop engine
├── briefer.py    — Morning briefing synthesizer
├── jobs.py       — Job spec CRUD and templates
└── models.py     — Ollama HTTP client
```

### Error Handling Philosophy

- **Ollama connection failure**: forge checks `is_available()` before starting and sets `ForgeState.ERROR` if unreachable
- **Job execution failure**: individual job failures are caught and logged; the forge continues to the next job
- **Job timeout**: enforced via `max_duration_seconds` at the session level
- **Disk full**: artifact `save()` will raise `OSError`; the forge catches it as a job failure
- **Signal handling**: SIGINT/SIGTERM trigger graceful shutdown after the current iteration

### Running Tests

```bash
pytest -v              # full suite
pytest tests/test_forge.py -v   # single module
pytest -k "dry_run"    # pattern match
```

Tests use `pytest`, `unittest.mock`, and `tmp_path` fixtures. No real Ollama connection needed.

## Adding a New Command

1. Add the command function in `cli.py` with `@main.command()`
2. Add any new logic in the appropriate module (`forge.py`, `briefer.py`, etc.)
3. Add tests in `tests/`
4. Update the README with usage and options table

## Submitting Changes

1. Feature branch: `git checkout -b feat/your-feature`
2. Run `pytest` and ensure all tests pass
3. Write clear commit messages (conventional commits preferred)
4. Open a PR with a description of what changed and why

## Reporting Bugs

Include:
- Python version (`python --version`)
- Ollama version and model used
- The exact `forge` command that failed
- Full error traceback
- Contents of `.forge-state.json` if it exists

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
