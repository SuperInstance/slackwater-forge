"""
Job spec manager — defines and manages what the forge works on.

Job specs are JSON files that define topics, models, schedules, and priorities
for overnight forge runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class JobType(str, Enum):
    CODE_REVIEW = "code_review"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    BRAINSTORM = "brainstorm"
    CUSTOM = "custom"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class JobSpec(BaseModel):
    """A single job specification for the forge."""
    id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Human-readable name")
    type: JobType = JobType.CUSTOM
    prompt: str = Field("", description="Prompt template for this job")
    system_prompt: str = Field("", description="System prompt for this job")
    model: str = Field("granite3.1-dense:2b", description="Ollama model to use")
    priority: Priority = Priority.MEDIUM
    token_budget: int = Field(50000, description="Max tokens per iteration")
    max_iterations: int = Field(1, description="Max forge iterations for this job")
    output_format: str = Field("markdown", description="Output format")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    enabled: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    options: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7},
        description="Additional Ollama options",
    )


class ForgeSession(BaseModel):
    """A complete forge session definition — the overnight job spec."""
    name: str = "overnight"
    description: str = ""
    models: list[str] = Field(default_factory=lambda: ["granite3.1-dense:2b"])
    start_time: str = ""
    end_time: str = ""
    jobs: list[JobSpec] = Field(default_factory=list)
    global_options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "ForgeSession":
        """Load a forge session from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Job spec file not found: {path}")
        data = json.loads(path.read_text())
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save the forge session to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2))

    def add_job(self, job: JobSpec) -> None:
        self.jobs.append(job)

    def remove_job(self, job_id: str) -> bool:
        original_len = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.id != job_id]
        return len(self.jobs) < original_len

    def get_job(self, job_id: str) -> JobSpec | None:
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def enabled_jobs(self) -> list[JobSpec]:
        """Return only enabled jobs, sorted by priority."""
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        return sorted(
            [j for j in self.jobs if j.enabled],
            key=lambda j: priority_order.get(j.priority, 99),
        )


class JobManager:
    """
    Manages job specs on disk.

    Directory structure:
        jobs/
          active/
            overnight.json
            code-audit.json
          archive/
            2026-08-04-overnight.json
          templates/
            code-review.json
    """

    def __init__(self, base_dir: str | Path = "jobs") -> None:
        self.base_dir = Path(base_dir)
        self.active_dir = self.base_dir / "active"
        self.archive_dir = self.base_dir / "archive"
        self.template_dir = self.base_dir / "templates"
        for d in (self.active_dir, self.archive_dir, self.template_dir):
            d.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[Path]:
        """List all active session files."""
        return sorted(self.active_dir.glob("*.json"))

    def load_session(self, name: str) -> ForgeSession:
        """Load a session by name (without .json extension)."""
        path = self.active_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{name}' not found at {path}")
        return ForgeSession.from_file(path)

    def save_session(self, session: ForgeSession, name: str | None = None) -> Path:
        """Save a session. Returns the path it was saved to."""
        if name:
            session.name = name
        path = self.active_dir / f"{session.name}.json"
        session.save(path)
        return path

    def archive_session(self, name: str) -> Path | None:
        """Move a session from active to archive."""
        src = self.active_dir / f"{name}.json"
        if not src.exists():
            return None
        timestamp = datetime.now().strftime("%Y-%m-%d")
        dst = self.archive_dir / f"{timestamp}-{name}.json"
        src.rename(dst)
        return dst

    def delete_session(self, name: str) -> bool:
        """Permanently delete a session."""
        path = self.active_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def create_session(
        self,
        name: str,
        description: str = "",
        models: list[str] | None = None,
    ) -> ForgeSession:
        """Create a new empty session."""
        session = ForgeSession(
            name=name,
            description=description,
            models=models or ["granite3.1-dense:2b"],
        )
        self.save_session(session)
        return session

    def create_job(
        self,
        job_id: str,
        name: str,
        prompt: str,
        job_type: JobType = JobType.CUSTOM,
        model: str = "granite3.1-dense:2b",
        priority: Priority = Priority.MEDIUM,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> JobSpec:
        """Create a new JobSpec with sensible defaults."""
        return JobSpec(
            id=job_id,
            name=name,
            prompt=prompt,
            type=job_type,
            model=model,
            priority=priority,
            system_prompt=system_prompt,
            **kwargs,
        )


# Built-in templates for common forge tasks
BUILTIN_TEMPLATES: dict[str, dict[str, Any]] = {
    "code-review": {
        "id": "code_review",
        "name": "Code Review",
        "type": "code_review",
        "prompt": (
            "Review the following code for bugs, performance issues, and best practices. "
            "Provide specific, actionable feedback with line references.\n\n"
            "Code to review:\n{code}"
        ),
        "system_prompt": "You are an expert code reviewer. Be thorough but concise.",
        "model": "granite3.1-dense:2b",
        "priority": "high",
        "output_format": "markdown",
    },
    "creative-writing": {
        "id": "creative_writing",
        "name": "Creative Writing",
        "type": "creative_writing",
        "prompt": (
            "Write a {length} {form} about {topic}. "
            "Match the tone: {tone}. Be vivid and specific."
        ),
        "system_prompt": "You are a skilled creative writer with a gift for atmosphere.",
        "model": "granite3.1-dense:2b",
        "priority": "medium",
        "output_format": "markdown",
    },
    "research": {
        "id": "research",
        "name": "Research Deep-Dive",
        "type": "research",
        "prompt": (
            "Research the following topic and provide a structured analysis "
            "with key findings, trade-offs, and recommendations.\n\n"
            "Topic: {topic}\nDepth: {depth}"
        ),
        "system_prompt": (
            "You are a research analyst. Be objective, cite specifics, "
            "and distinguish facts from inferences."
        ),
        "model": "granite3.1-dense:2b",
        "priority": "medium",
        "output_format": "markdown",
    },
    "brainstorm": {
        "id": "brainstorm",
        "name": "Brainstorm Session",
        "type": "brainstorm",
        "prompt": (
            "Generate {count} creative ideas for: {topic}. "
            "For each idea, provide a one-line summary and a brief rationale."
        ),
        "system_prompt": "You are a creative strategist. Think divergently.",
        "model": "granite3.1-dense:2b",
        "priority": "low",
        "output_format": "markdown",
    },
    "documentation": {
        "id": "documentation",
        "name": "Documentation Writer",
        "type": "documentation",
        "prompt": (
            "Write clear, comprehensive documentation for the following.\n\n"
            "Subject: {subject}\nAudience: {audience}\nFormat: {format}"
        ),
        "system_prompt": (
            "You are a technical writer. Be clear, structured, and complete."
        ),
        "model": "granite3.1-dense:2b",
        "priority": "medium",
        "output_format": "markdown",
    },
}
