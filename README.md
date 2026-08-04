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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                              │
│  forge run · forge brief · forge job · forge status · forge test │
└──────────┬──────────────────┬───────────────────┬──────────────┘
           │                  │                   │
           ▼                  ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Forge Engine   │ │     Briefer      │ │   JobManager     │
│    (forge.py)    │ │  (briefer.py)    │ │    (jobs.py)     │
│                  │ │                  │ │                  │
│ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │
│ │ Round-robin  │ │ │ │ Load .json   │ │ │ │ Session CRUD │ │
│ │ job loop     │ │ │ │ artifacts    │ │ │ │ Template mgr │ │
│ │              │ │ │ │              │ │ │ │              │ │
│ │ Time/iter    │ │ │ │ AI summarize │ │ │ │ jobs/active/ │ │
│ │ limits       │ │ │ │ (or offline) │ │ │ │ jobs/archive/│ │
│ │              │ │ │ │              │ │ │ │ jobs/templts/│ │
│ │ Signal-safe  │ │ │ │ MD + HTML    │ │ │ └──────────────┘ │
│ │ shutdown     │ │ │ │ output       │ │ │                  │
│ └──────┬───────┘ │ │ └──────────────┘ │ └──────────────────┘
│        │         │ └──────────────────┘
│        ▼         │
│ ┌──────────────┐ │
│ │  Artifacts   │ │
│ │  saved to    │ │
│ │  disk as     │ │
│ │  .md + .json │ │
│ └──────────────┘ │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│     Ollama Client (models.py)        │
│                                      │
│  POST /api/generate   (text gen)     │
│  POST /api/chat       (chat turn)    │
│  GET  /api/tags       (model list)   │
│  POST /api/pull       (model pull)   │
│                                      │
│  Connection: http://localhost:11434  │
└──────────────────────────────────────┘
```

### Data Flow

```
Job Session (JSON)          Forge Loop              Artifacts              Briefing
┌─────────────────┐        ┌──────────┐         ┌─────────────┐       ┌─────────────┐
│ {               │ ─────▶ │          │ ─────▶  │ *.md        │ ────▶ │ briefing-   │
│   "jobs": [     │        │  Round-  │         │ *.json      │       │ YYYY-MM-DD  │
│     {...},      │        │  robin   │         │             │       │   .md       │
│     {...}       │        │          │         │             │       │   .html     │
│   ]             │        │          │         │             │       │             │
│ }               │        └──────────┘         └─────────────┘       └─────────────┘
└─────────────────┘              │                                          ▲
                                 │                                          │
                          Ollama API calls ───────────────────────────────┘
                          (generate / chat)                          (AI summary)
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

## Full Job Spec Format

Sessions are JSON files stored in `jobs/active/`:

```json
{
  "name": "overnight",
  "description": "Overnight code audit + creative writing",
  "models": ["granite3.1-dense:2b"],
  "global_options": {},
  "jobs": [
    {
      "id": "lua_audit",
      "name": "Lua Code Audit",
      "type": "code_review",
      "prompt": "Review this Lua code for bugs:\n{code}",
      "system_prompt": "You are an expert Roblox Lua reviewer.",
      "model": "granite3.1-dense:2b",
      "priority": "high",
      "token_budget": 50000,
      "max_iterations": 5,
      "output_format": "markdown",
      "tags": ["roblox", "audit"],
      "enabled": true,
      "options": {
        "temperature": 0.3,
        "num_ctx": 8192
      }
    },
    {
      "id": "lore",
      "name": "Harbor Vignettes",
      "type": "creative_writing",
      "prompt": "Write a short story about life in Slackwater harbor.",
      "system_prompt": "You are a skilled creative writer.",
      "model": "granite3.1-dense:2b",
      "priority": "low",
      "max_iterations": 3,
      "options": {
        "temperature": 0.9
      }
    }
  ]
}
```

### Job Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | *(required)* | Unique job identifier |
| `name` | string | *(required)* | Human-readable name |
| `type` | enum | `custom` | `code_review`, `creative_writing`, `research`, `analysis`, `documentation`, `brainstorm`, `custom` |
| `prompt` | string | `""` | Prompt template. `{iteration}` is replaced with the iteration number |
| `system_prompt` | string | `""` | System prompt for the model |
| `model` | string | `granite3.1-dense:2b` | Ollama model name |
| `priority` | enum | `medium` | `critical`, `high`, `medium`, `low` — higher priority jobs run first |
| `token_budget` | int | `50000` | Max tokens per iteration (informational) |
| `max_iterations` | int | `1` | How many times this job runs before it's "completed" |
| `output_format` | string | `markdown` | Output format hint |
| `tags` | string[] | `[]` | Categorization tags |
| `enabled` | bool | `true` | If false, the job is skipped |
| `options` | object | `{"temperature": 0.7}` | Passed directly to Ollama's `options` field |

## Example Workflows

### Workflow 1: Overnight Code Audit

```bash
# Create a session for auditing a Roblox project
forge job create -n code-audit -m granite3.1-dense:2b

forge job add code-audit \
  --id "init-audit" \
  --name "Init Module Audit" \
  --type code_review \
  --priority high \
  --prompt "Review this Roblox init.lua for bugs, edge cases, and performance issues: $(cat src/init.lua)" \
  --max-iterations 3

forge job add code-audit \
  --id "patterns-audit" \
  --name "Patterns Module Audit" \
  --type code_review \
  --priority medium \
  --prompt "Review this Patterns.lua module: $(cat src/Patterns.lua)" \
  --max-iterations 2

# Run overnight
forge run -s code-audit --duration 8h

# Morning briefing
forge brief --format md --format html --open
```

### Workflow 2: Creative Worldbuilding

```bash
forge job create -n worldbuilding -m qwen2.5:7b

forge job add worldbuilding \
  --id "npc-backstory" \
  --name "NPC Backstories" \
  --type creative_writing \
  --priority medium \
  --prompt "Write a detailed backstory for an NPC named {name} who lives in a coastal fishing village." \
  --max-iterations 10

forge job add worldbuilding \
  --id "location-desc" \
  --name "Location Descriptions" \
  --type creative_writing \
  --priority low \
  --prompt "Describe a {location_type} in a fantasy harbor town. Include sensory details, inhabitants, and a hidden secret." \
  --max-iterations 5

forge run -s worldbuilding --duration 4h
forge brief -r "Worldbuilder Team"
```

### Workflow 3: Research Deep-Dive

```json
// jobs/active/research.json
{
  "name": "research",
  "description": "Research session for architecture decisions",
  "models": ["granite3.1-dense:2b"],
  "jobs": [
    {
      "id": "lua-patterns",
      "name": "Lua Design Patterns",
      "type": "research",
      "prompt": "Research common design patterns used in Roblox Lua game development. Focus on: module pattern, observer pattern, service locator, state machines. For each, provide a code example and trade-offs.",
      "model": "granite3.1-dense:2b",
      "priority": "high",
      "max_iterations": 5,
      "options": {"temperature": 0.3}
    },
    {
      "id": "perf-analysis",
      "name": "Performance Analysis Techniques",
      "type": "analysis",
      "prompt": "Analyze the best practices for profiling and optimizing Roblox game performance. Cover: Luau optimizations, memory management, RemoteEvent batching, and DrawCall reduction.",
      "model": "granite3.1-dense:2b",
      "priority": "medium",
      "max_iterations": 3
    }
  ]
}
```

```bash
forge run -s research --iterations 8
forge brief --no-ai  # offline summary, no AI needed
```

## Output Structure

```
forge-output/
├── .forge-state.json              # running state
├── 2026-08-04-lua-audit-000.md    # artifacts (markdown)
├── 2026-08-04-lua-audit-000.json  # artifact metadata
├── 2026-08-04-lore-001.md
├── 2026-08-04-lore-001.json
├── briefing-2026-08-05.md         # morning briefing (markdown)
└── briefing-2026-08-05.html       # morning briefing (styled HTML)
```

## Module Reference

```
slackwater_forge/
├── __init__.py     — package metadata, version
├── cli.py          — Click CLI: run, brief, job, status, models, test
├── forge.py        — Forge engine: job loop, artifact saving, state tracking
│   ├── Forge       — main engine class
│   ├── Artifact    — single output artifact (text + metadata)
│   ├── ForgeStats  — running statistics
│   └── ForgeState  — IDLE → RUNNING → STOPPED state machine
├── briefer.py      — Briefer: reads artifacts, generates summary
│   ├── Briefer     — briefing generator (AI + offline modes)
│   └── HTML_TEMPLATE — Jinja2 template for HTML briefings
├── jobs.py         — Job management
│   ├── JobSpec     — Pydantic model for individual jobs
│   ├── ForgeSession — Pydantic model for a session of jobs
│   ├── JobManager  — disk-based session CRUD
│   └── BUILTIN_TEMPLATES — 5 pre-built job templates
└── models.py       — Ollama HTTP client
    ├── OllamaClient — sync client with context manager support
    ├── ModelInfo   — model metadata
    └── GenerateResult — generation output with token stats
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `localhost` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `FORGE_OUTPUT` | `forge-output` | Output directory |

## Troubleshooting

### "Ollama is not available"

**Symptom:** `forge run` exits immediately with "Ollama is not available at http://localhost:11434"

**Causes & Fixes:**
1. **Ollama not running** — start it: `ollama serve`
2. **Wrong host/port** — check with: `curl http://localhost:11434/api/tags`
3. **Firewall blocking** — ensure port 11434 is open: `sudo ufw allow 11434`
4. **Remote Ollama** — set `OLLAMA_HOST` and `OLLAMA_PORT` env vars

### "Missing models"

**Symptom:** Forge warns about missing models before starting.

**Fix:** Pull the required models:
```bash
ollama pull granite3.1-dense:2b
# Or override all jobs to use an available model:
forge run -s overnight -m llama3.1:8b
```

### Job fails mid-iteration

**Symptom:** Individual job fails but the forge continues.

**Behavior:** The forge logs the error, increments the error counter, and moves to the next job. Check the forge output for the error message. Common causes:
- Model ran out of context (`num_ctx` too small — increase in job `options`)
- Model returned empty response (try a different model or adjust `temperature`)
- Network hiccup to Ollama (the forge will retry on the next cycle)

### Disk full

**Symptom:** Artifact save fails with `OSError: [Errno 28] No space left on device`

**Behavior:** The forge catches the error as a job failure and continues. If the disk is truly full, subsequent saves will also fail and the forge will eventually exhaust all jobs.

**Fix:** Clear old artifacts:
```bash
# Remove artifacts older than 7 days
find forge-output/ -name "*.md" -mtime +7 -delete
find forge-output/ -name "*.json" -mtime +7 -delete
```

### SIGINT/SIGTERM handling

The forge registers signal handlers for SIGINT (Ctrl-C) and SIGTERM. When either is received:
1. The current iteration completes normally (the artifact is saved)
2. The forge state is set to `STOPPED`
3. State is persisted to `.forge-state.json`
4. The forge loop exits cleanly

This means pressing Ctrl-C during a long generation will wait for that generation to finish before stopping. If you need to force-kill, press Ctrl-C twice rapidly.

### Briefing shows "No artifacts found"

**Cause:** The output directory has no `.json` metadata files.

**Fix:**
1. Check you're pointing at the right directory: `--output forge-output/`
2. Run a dry-run first to verify artifacts are created: `forge run --dry-run -s overnight -i 3`
3. Check `.forge-state.json` for the last known state

## Why "Slackwater"?

Slackwater is the calm period when the tide turns — the in-between time. The forge runs during your slackwater (overnight) and produces value by morning. Also: it works on a laptop GPU, not a data center. It's the small-scale, craft approach to AI-assisted productivity.

## License

MIT — see [LICENSE](LICENSE)
