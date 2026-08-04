"""
Core forge loop — the overnight production line.

Executes jobs against Ollama models, saves artifacts, tracks state.
"""

from __future__ import annotations

import json
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .jobs import ForgeSession, JobSpec, JobType, Priority
from .models import OllamaClient, GenerateResult

logger = logging.getLogger("slackwater_forge")


class ForgeState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Artifact:
    """A single output artifact from a forge iteration."""
    job_id: str
    job_name: str
    job_type: str
    model: str
    priority: str
    text: str
    prompt: str
    iteration: int
    tokens_generated: int
    tokens_per_second: float
    elapsed_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False
    confidence: float = 0.0

    def slug(self) -> str:
        """Filesystem-safe name for this artifact."""
        safe_id = re.sub(r"[^a-z0-9-]", "-", self.job_id.lower())
        return f"{self.timestamp[:10]}-{safe_id}-{self.iteration:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "job_type": self.job_type,
            "model": self.model,
            "priority": self.priority,
            "text": self.text,
            "prompt": self.prompt,
            "iteration": self.iteration,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "confidence": self.confidence,
        }

    def save(self, output_dir: Path) -> Path:
        """Save artifact as both .md and .json (metadata)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = self.slug()

        md_path = output_dir / f"{slug}.md"
        md_path.write_text(self.text)

        json_path = output_dir / f"{slug}.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2))

        return md_path


@dataclass
class ForgeStats:
    """Running statistics for a forge session."""
    start_time: str = ""
    iterations: int = 0
    artifacts_produced: int = 0
    total_tokens: int = 0
    errors: int = 0
    jobs_completed: list[str] = field(default_factory=list)
    models_used: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "iterations": self.iterations,
            "artifacts_produced": self.artifacts_produced,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "jobs_completed": list(self.jobs_completed),
            "models_used": list(self.models_used),
            "elapsed_seconds": self._elapsed_seconds(),
        }

    def _elapsed_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        try:
            start = datetime.fromisoformat(self.start_time)
            return (datetime.now() - start).total_seconds()
        except (ValueError, TypeError):
            return 0.0


class Forge:
    """
    The core forge engine.

    Runs jobs in a loop, calling Ollama for each, saving artifacts.
    Supports:
      - Time-limited runs (e.g., overnight 8 hours)
      - Iteration-limited runs (e.g., 50 iterations)
      - Continuous runs until stopped
      - Round-robin through enabled jobs
    """

    def __init__(
        self,
        output_dir: str | Path = "forge-output",
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        state_file: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_file) if state_file else self.output_dir / ".forge-state.json"
        self.client = OllamaClient(host=ollama_host, port=ollama_port)
        self.state: ForgeState = ForgeState.IDLE
        self.stats = ForgeStats()
        self._stop_requested = False
        self._on_iteration: Callable[[Artifact], None] | None = None

    def _save_state(self) -> None:
        """Persist current state for status checks."""
        state_data = {
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "saved_at": datetime.now().isoformat(),
        }
        self.state_file.write_text(json.dumps(state_data, indent=2, default=str))

    def _load_state(self) -> dict[str, Any] | None:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def request_stop(self) -> None:
        """Request the forge to stop after the current iteration."""
        self._stop_requested = True
        self.state = ForgeState.STOPPED

    def _setup_signal_handlers(self) -> None:
        """Handle Ctrl-C gracefully."""
        def handler(signum: int, frame: Any) -> None:
            logger.info("Stop signal received, finishing current iteration...")
            self.request_stop()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _format_prompt(self, job: JobSpec, iteration: int) -> str:
        """Format the job's prompt template with iteration context."""
        prompt = job.prompt
        # Support {iteration} placeholder
        if "{iteration}" in prompt:
            prompt = prompt.format(iteration=iteration)
        return prompt

    def _execute_job(self, job: JobSpec, iteration: int) -> Artifact:
        """Execute a single job against Ollama."""
        prompt = self._format_prompt(job, iteration)
        logger.info(
            "Executing job '%s' (iteration %d) with model '%s'",
            job.name, iteration, job.model,
        )

        result = self.client.generate(
            model=job.model,
            prompt=prompt,
            system=job.system_prompt or "",
            options=job.options,
        )

        artifact = Artifact(
            job_id=job.id,
            job_name=job.name,
            job_type=job.type.value,
            model=result.model,
            priority=job.priority.value,
            text=result.text,
            prompt=prompt,
            iteration=iteration,
            tokens_generated=result.eval_count,
            tokens_per_second=result.tokens_per_second,
            elapsed_seconds=result.elapsed_seconds,
        )

        # Save artifact
        saved_path = artifact.save(self.output_dir)
        logger.info("Artifact saved: %s (%d tokens, %.1f tok/s)",
                     saved_path, result.eval_count, result.tokens_per_second)

        # Callback
        if self._on_iteration:
            self._on_iteration(artifact)

        # Update stats
        self.stats.artifacts_produced += 1
        self.stats.total_tokens += result.eval_count
        self.stats.models_used.add(result.model)

        return artifact

    def run_session(
        self,
        session: ForgeSession,
        max_iterations: int | None = None,
        max_duration_seconds: int | None = None,
        continuous: bool = False,
        dry_run: bool = False,
        on_iteration: Callable[[Artifact], None] | None = None,
    ) -> ForgeStats:
        """
        Run a forge session.

        Args:
            session: The forge session with jobs to execute.
            max_iterations: Maximum total iterations across all jobs.
            max_duration_seconds: Time limit in seconds (e.g., 28800 for 8 hours).
            continuous: If True, loop forever until stopped or signal.
            dry_run: If True, don't actually call Ollama — just log what would happen.
            on_iteration: Optional callback after each artifact.

        Returns:
            Final ForgeStats.
        """
        self._on_iteration = on_iteration
        self._stop_requested = False
        self._setup_signal_handlers()

        enabled_jobs = session.enabled_jobs()
        if not enabled_jobs:
            logger.warning("No enabled jobs in session '%s'", session.name)
            return self.stats

        # Check Ollama availability
        if not dry_run and not self.client.is_available():
            logger.error("Ollama is not available at %s", self.client.base_url)
            self.state = ForgeState.ERROR
            self._save_state()
            return self.stats

        self.state = ForgeState.RUNNING
        self.stats.start_time = datetime.now().isoformat()
        start_ts = time.monotonic()
        logger.info(
            "Starting forge session '%s' with %d jobs (%s mode)",
            session.name,
            len(enabled_jobs),
            "continuous" if continuous else "bounded",
        )

        try:
            iteration_counter = 0
            job_index = 0

            while not self._stop_requested:
                # Check limits
                if not continuous:
                    if max_iterations and iteration_counter >= max_iterations:
                        logger.info("Reached max_iterations=%d", max_iterations)
                        break
                    if max_duration_seconds:
                        elapsed = time.monotonic() - start_ts
                        if elapsed >= max_duration_seconds:
                            logger.info("Reached max_duration=%ds", max_duration_seconds)
                            break

                # Round-robin through jobs
                job = enabled_jobs[job_index % len(enabled_jobs)]

                # Check per-job iteration limit
                job_iterations = [
                    a for a in self._iter_artifacts()
                    if a.get("job_id") == job.id
                ]
                if len(job_iterations) >= job.max_iterations and job.max_iterations > 0:
                    if job.id not in self.stats.jobs_completed:
                        self.stats.jobs_completed.append(job.id)
                        logger.info("Job '%s' completed (%d/%d iterations)",
                                    job.name, len(job_iterations), job.max_iterations)

                    # Check if all jobs are completed
                    all_done = all(
                        len([a for a in self._iter_artifacts()
                             if a.get("job_id") == j.id]) >= j.max_iterations
                        for j in enabled_jobs
                        if j.max_iterations > 0
                    )
                    if all_done and not continuous:
                        logger.info("All jobs completed!")
                        break

                    job_index += 1
                    if job_index % len(enabled_jobs) == 0:
                        # All jobs exhausted or completed
                        remaining = [
                            j for j in enabled_jobs
                            if j.max_iterations == 0 or
                            len([a for a in self._iter_artifacts()
                                 if a.get("job_id") == j.id]) < j.max_iterations
                        ]
                        if not remaining:
                            break
                        time.sleep(0.1)  # Prevent tight loop
                    continue

                # Execute the job
                if dry_run:
                    logger.info("[DRY RUN] Would execute job '%s' iteration %d",
                                job.name, iteration_counter)
                else:
                    try:
                        self._execute_job(job, iteration_counter)
                    except Exception as e:
                        logger.error("Job '%s' failed: %s", job.name, e)
                        self.stats.errors += 1

                self.stats.iterations += 1
                iteration_counter += 1
                job_index += 1

                self._save_state()

                # Brief pause between calls to let GPU breathe
                if not dry_run:
                    time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.state = ForgeState.STOPPED
            self._save_state()
            self.client.close()

        elapsed = time.monotonic() - start_ts
        logger.info(
            "Forge session complete: %d iterations, %d artifacts, %d errors, %.1fs elapsed",
            self.stats.iterations,
            self.stats.artifacts_produced,
            self.stats.errors,
            elapsed,
        )

        return self.stats

    def _iter_artifacts(self) -> list[dict[str, Any]]:
        """Quick scan of saved artifacts (JSON metadata only)."""
        artifacts = []
        for f in self.output_dir.glob("*.json"):
            if f.name.startswith("."):
                continue
            try:
                data = json.loads(f.read_text())
                artifacts.append(data)
            except (json.JSONDecodeError, IOError):
                continue
        return artifacts

    def get_status(self) -> dict[str, Any]:
        """Get current forge status for display."""
        saved = self._load_state()
        artifacts = self._iter_artifacts()
        models_used = set(a.get("model", "") for a in artifacts)

        return {
            "state": self.state.value if self.state != ForgeState.IDLE
                     else saved.get("state", "idle") if saved else "idle",
            "output_dir": str(self.output_dir),
            "artifact_count": len(artifacts),
            "models_used": sorted(m for m in models_used if m),
            "ollama_available": self.client.is_available(),
            "ollama_url": self.client.base_url,
            "stats": saved.get("stats") if saved else self.stats.to_dict(),
            "last_updated": saved.get("saved_at", "") if saved else "",
        }
