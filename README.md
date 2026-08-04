# 🔥 Slackwater Forge

**Overnight GPU production line that produces a morning briefing.**

Slackwater Forge treats a local GPU (via [Ollama](https://ollama.ai)) as a production line. You define jobs, start the forge, and let it run overnight. In the morning, it synthesizes all the artifacts into a structured briefing.

```
$ forge run --session overnight --duration 8h
$ forge brief --format md --format html
```

## Features

- **Works with ANY Ollama model** — Granite, Qwen, Llama, Mistral, Phi, anything
- **Job spec system** — define what the forge works on (code review, creative writing, research, etc.)
- **Artifact tracking** — every output saved with metadata (model, tokens, timing)
- **Morning briefing** — AI-synthesized summary with priorities, findings, and recommendations
- **Offline mode** — generate briefings without Ollama (metadata-only synthesis)
- **Beautiful output** — markdown + styled HTML briefings
- **Cost: $0** — entirely local, no API keys, no cloud

## Install

```bash
git clone https://github.com/SuperInstance/slackwater-forge.git
cd slackwater-forge
pip install -e ".[dev]"
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- At least one pulled model: `ollama pull granite3.1-dense:2b`

## Quick Start

### 1. Check your setup
```bash
forge models          # list available Ollama models
forge test            # quick connectivity test
forge status          # current forge state
```

### 2. Create a job session
```bash
# Create session with built-in templates
forge job create --name overnight --template -m granite3.1-dense:2b

# Add custom jobs
forge job add overnight \
  --id "lua-audit" \
  --name "Lua Code Audit" \
  --type code_review \
  --priority high \
  --prompt "Review this Roblox Lua code for bugs: {code}" \
  --max-iterations 5

# View the session
forge job show overnight
```

### 3. Run the forge
```bash
# Overnight (8 hours)
forge run --session overnight --duration 8h

# Fixed iterations
forge run --session overnight --iterations 50

# Continuous (Ctrl-C to stop)
forge run --session overnight --continuous

# Dry run (no Ollama calls)
forge run --session overnight --dry-run
```

### 4. Generate the morning briefing
```bash
# Markdown
forge brief --format md

# Markdown + HTML (opens in browser)
forge brief --format md --format html --open

# Offline mode (no AI synthesis)
forge brief --no-ai

# Specify model for AI summary
forge brief -m qwen2.5:7b
```

## Commands

### `forge run`
Starts the overnight loop.

| Option | Description |
|--------|-------------|
| `-s, --session` | Session name from saved job specs |
| `-m, --model` | Override model for all jobs |
| `-i, --iterations` | Max total iterations |
| `-d, --duration` | Time limit (`8h`, `30m`, `3600s`) |
| `-c, --continuous` | Run until stopped |
| `--dry-run` | Don't call Ollama |

### `forge brief`
Generates a morning briefing.

| Option | Description |
|--------|-------------|
| `-m, --model` | Model for AI summary |
| `-r, --recipient` | Recipient name |
| `-f, --format` | Output format: `md`, `html` |
| `--no-ai` | Offline synthesis (no Ollama) |
| `--open` | Open HTML in browser |

### `forge job`
Manages job specs.

```bash
forge job create -n overnight --template
forge job list
forge job show <name>
forge job add <session> --id <id> --name <name> --prompt "..."
forge job remove <session> <job-id>
forge job delete <session>
```

### `forge status`
Shows forge state, GPU status, artifact count, and available models.

### `forge models`
Lists all Ollama models with size, quantization, and family info.

### `forge test`
Quick connectivity test — sends a prompt to verify the pipeline works.

## Job Spec Format

Sessions are JSON files in `jobs/active/`:

```json
{
  "name": "overnight",
  "description": "Overnight code audit + creative writing",
  "models": ["granite3.1-dense:2b"],
  "jobs": [
    {
      "id": "lua_audit",
      "name": "Lua Code Audit",
      "type": "code_review",
      "prompt": "Review this Lua code for bugs:\n{code}",
      "system_prompt": "You are an expert Roblox Lua reviewer.",
      "model": "granite3.1-dense:2b",
      "priority": "high",
      "max_iterations": 5,
      "options": {"temperature": 0.3}
    },
    {
      "id": "lore",
      "name": "Harbor Vignettes",
      "type": "creative_writing",
      "prompt": "Write a short story about life in Slackwater harbor.",
      "model": "granite3.1-dense:2b",
      "priority": "low",
      "max_iterations": 3
    }
  ]
}
```

## Output Structure

```
forge-output/
├── .forge-state.json              # running state
├── 2026-08-04-lua-audit-000.md    # artifacts
├── 2026-08-04-lua-audit-000.json  # metadata
├── 2026-08-04-lore-001.md
├── 2026-08-04-lore-001.json
├── briefing-2026-08-05.md         # morning briefing
└── briefing-2026-08-05.html
```

## Architecture

```
slackwater_forge/
├── cli.py        # Click CLI (forge run, brief, job, status, models, test)
├── forge.py      # Core loop — Ollama calls, artifact saving, state tracking
├── briefer.py    # Briefing synthesizer — reads artifacts, generates summary
├── jobs.py       # Job spec manager — session/job CRUD, templates
└── models.py     # Ollama API client — generate, chat, list models
```

## Why "Slackwater"?

Slackwater is the calm period when the tide turns — the in-between time. The forge runs during your slackwater (overnight) and produces value by morning. Also: it works on a laptop GPU, not a data center. It's the small-scale, craft approach to AI-assisted productivity.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `localhost` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `FORGE_OUTPUT` | `forge-output` | Output directory |

## License

MIT — see [LICENSE](LICENSE)
