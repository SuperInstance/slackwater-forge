"""Tests for the job spec manager."""
import json
import tempfile
from pathlib import Path

import pytest

from slackwater_forge.jobs import (
    JobManager,
    JobSpec,
    JobType,
    Priority,
    ForgeSession,
    BUILTIN_TEMPLATES,
)


class TestJobSpec:
    def test_create_minimal(self):
        spec = JobSpec(id="test", name="Test Job", prompt="Hello")
        assert spec.id == "test"
        assert spec.name == "Test Job"
        assert spec.type == JobType.CUSTOM
        assert spec.priority == Priority.MEDIUM
        assert spec.enabled is True

    def test_create_full(self):
        spec = JobSpec(
            id="code1",
            name="Code Review",
            prompt="Review this: {code}",
            type=JobType.CODE_REVIEW,
            model="llama3:8b",
            priority=Priority.HIGH,
            system_prompt="You are a reviewer.",
            max_iterations=5,
            tags=["python", "urgent"],
        )
        assert spec.type == JobType.CODE_REVIEW
        assert spec.priority == Priority.HIGH
        assert spec.max_iterations == 5
        assert "python" in spec.tags


class TestForgeSession:
    def test_create_empty(self):
        session = ForgeSession(name="test")
        assert session.name == "test"
        assert session.jobs == []

    def test_add_remove_job(self):
        session = ForgeSession(name="test")
        job = JobSpec(id="j1", name="Job 1", prompt="test")
        session.add_job(job)
        assert len(session.jobs) == 1

        assert session.remove_job("j1") is True
        assert len(session.jobs) == 0
        assert session.remove_job("nonexistent") is False

    def test_get_job(self):
        session = ForgeSession(name="test")
        job = JobSpec(id="j1", name="Job 1", prompt="test")
        session.add_job(job)
        assert session.get_job("j1") == job
        assert session.get_job("nope") is None

    def test_enabled_jobs_sorted_by_priority(self):
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="low", name="Low", prompt="x", priority=Priority.LOW))
        session.add_job(JobSpec(id="high", name="High", prompt="x", priority=Priority.HIGH))
        session.add_job(JobSpec(id="crit", name="Crit", prompt="x", priority=Priority.CRITICAL))
        session.add_job(JobSpec(id="disabled", name="Dis", prompt="x", enabled=False))

        enabled = session.enabled_jobs()
        assert len(enabled) == 3
        assert enabled[0].id == "crit"
        assert enabled[1].id == "high"
        assert enabled[2].id == "low"

    def test_save_and_load(self, tmp_path):
        session = ForgeSession(name="test", description="Test session")
        session.add_job(JobSpec(id="j1", name="Job 1", prompt="Do {thing}"))
        session.add_job(JobSpec(id="j2", name="Job 2", prompt="Analyze", type=JobType.ANALYSIS))

        path = tmp_path / "session.json"
        session.save(path)

        loaded = ForgeSession.from_file(path)
        assert loaded.name == "test"
        assert len(loaded.jobs) == 2
        assert loaded.jobs[0].id == "j1"
        assert loaded.jobs[1].type == JobType.ANALYSIS


class TestJobManager:
    def test_create_and_load(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        session = mgr.create_session(name="test", description="Test")
        assert session.name == "test"

        loaded = mgr.load_session("test")
        assert loaded.name == "test"

    def test_list_sessions(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session(name="s1")
        mgr.create_session(name="s2")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_archive_session(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session(name="test")
        archived = mgr.archive_session("test")
        assert archived is not None
        assert archived.exists()
        assert not (mgr.active_dir / "test.json").exists()

    def test_delete_session(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session(name="test")
        assert mgr.delete_session("test") is True
        assert mgr.delete_session("test") is False

    def test_load_nonexistent(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        with pytest.raises(FileNotFoundError):
            mgr.load_session("nonexistent")


class TestBuiltinTemplates:
    def test_all_templates_valid(self):
        for key, data in BUILTIN_TEMPLATES.items():
            spec = JobSpec(**data)
            assert spec.id
            assert spec.name
            assert spec.prompt
            assert spec.model

    def test_template_count(self):
        assert len(BUILTIN_TEMPLATES) >= 4
