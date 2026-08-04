"""
Slackwater Forge — Overnight GPU production line and morning briefing generator.

A CLI tool that treats a local GPU (via Ollama) as a production line that keeps
working overnight and produces a morning briefing from accumulated artifacts.
"""

__version__ = "0.1.0"
__all__ = ["cli", "forge", "briefer", "jobs", "models"]
