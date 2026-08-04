#!/usr/bin/env bash
#
# forge_demo.sh — Run a 5-minute mini Slackwater Forge session.
#
# Requirements:
#   - Ollama running on localhost:11434
#   - At least one model installed (e.g., `ollama pull granite3.1-dense:2b`)
#   - Python 3.12+ with the slackwater_forge package installed
#
# This script:
#   1. Checks Ollama is reachable
#   2. Creates a small forge session with 2 quick jobs
#   3. Runs the forge for ~5 minutes (or until jobs complete)
#   4. Generates a morning briefing from the artifacts
#   5. Prints the briefing to stdout
#
# Usage:
#   ./forge_demo.sh [--dry-run]
#
# --dry-run: Skip Ollama calls, just show the flow.

set -euo pipefail

OUTPUT_DIR="${FORGE_OUTPUT_DIR:-/tmp/forge-demo-$$}"
DURATION="${FORGE_DURATION:-300}"  # 5 minutes in seconds
MODEL="${FORGE_MODEL:-granite3.1-dense:2b}"
DRY_RUN=""

# ── Parse args ──────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run"; echo "[forge_demo] DRY RUN — no Ollama calls" ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"
      echo ""
      echo "Runs a 5-minute mini forge session."
      echo ""
      echo "Environment variables:"
      echo "  FORGE_OUTPUT_DIR  Output directory (default: /tmp/forge-demo-PID)"
      echo "  FORGE_DURATION    Run duration in seconds (default: 300)"
      echo "  FORGE_MODEL       Ollama model name (default: granite3.1-dense:2b)"
      exit 0
      ;;
  esac
done

echo "═" 60
echo "  🔥 Slackwater Forge — Mini Demo"
echo "═" 60
echo "  Output: $OUTPUT_DIR"
echo "  Model:  $MODEL"
echo "  Duration: ${DURATION}s"
echo ""

# ── Step 1: Check Ollama ────────────────────────────────────────
if [ -z "$DRY_RUN" ]; then
  echo "▸ Checking Ollama..."
  if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✗ Ollama not reachable at localhost:11434"
    echo "  Start it with: ollama serve"
    echo "  Or run with --dry-run to skip."
    exit 1
  fi
  echo "✓ Ollama is running"
fi

# ── Step 2: Create forge session ────────────────────────────────
echo ""
echo "▸ Creating forge session..."

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/session.json" << EOF
{
  "name": "mini-demo",
  "description": "5-minute demo: quick code tip + creative micro-story",
  "models": ["$MODEL"],
  "jobs": [
    {
      "id": "quick_tip",
      "name": "Luau Quick Tip",
      "type": "custom",
      "prompt": "Write a single paragraph Luau optimization tip. Be specific with code. Topic: efficient table iteration in Roblox.",
      "system_prompt": "You are a Roblox Luau expert. One paragraph, include a code snippet.",
      "model": "$MODEL",
      "priority": "high",
      "max_iterations": 2,
      "enabled": true,
      "options": {"temperature": 0.4}
    },
    {
      "id": "micro_story",
      "name": "Micro Story",
      "type": "creative_writing",
      "prompt": "Write a 100-word story about a lighthouse keeper who discovers a message in a bottle.",
      "system_prompt": "You are a flash fiction writer. Exactly 100 words.",
      "model": "$MODEL",
      "priority": "medium",
      "max_iterations": 2,
      "enabled": true,
      "options": {"temperature": 0.9}
    }
  ],
  "global_options": {}
}
EOF

echo "✓ Session saved to $OUTPUT_DIR/session.json"

# ── Step 3: Run the forge ───────────────────────────────────────
echo ""
echo "▸ Starting forge (${DURATION}s max)..."
echo ""

python3 -c "
import sys, json
from slackwater_forge.forge import Forge
from slackwater_forge.jobs import ForgeSession
from pathlib import Path

session = ForgeSession.from_file('$OUTPUT_DIR/session.json')
forge = Forge(output_dir='$OUTPUT_DIR')

stats = forge.run_session(
    session=session,
    max_duration_seconds=$DURATION,
    dry_run=${DRY_RUN:+True}${DRY_RUN:-False},
)

print()
print(f'✓ Forge complete: {stats.artifacts_produced} artifacts, '
      f'{stats.total_tokens} tokens, {stats.errors} errors')
" || {
  echo ""
  echo "⚠ Forge completed with issues (this is fine for demo purposes)"
}

# ── Step 4: Generate briefing ───────────────────────────────────
echo ""
echo "▸ Generating briefing..."
echo ""

python3 -c "
from slackwater_forge.briefer import Briefer

briefer = Briefer(output_dir='$OUTPUT_DIR')
briefing = briefer.generate(model='$MODEL', use_ai_summary=False)
md = briefer.to_markdown(briefing)
print(md)

# Save to file
briefer.save_briefing(briefing, formats=['md', 'html'])
" 2>/dev/null || {
  # Fallback: just list what was produced
  echo "─" 40
  echo "Artifacts in $OUTPUT_DIR:"
  ls -la "$OUTPUT_DIR"/*.md 2>/dev/null || echo "  (no markdown artifacts)"
  ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON artifacts)"
}

echo ""
echo "═" 60
echo "  ✓ Demo complete. Artifacts in: $OUTPUT_DIR"
echo "═" 60
