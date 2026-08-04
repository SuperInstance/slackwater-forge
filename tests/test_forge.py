"""Tests for the forge core engine."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slackwater_forge.forge import Artifact, Forge, ForgeState, ForgeStats
from slackwater_forge.jobs import ForgeSession, JobSpec, JobType, Priority


class TestArtifact:
    def test_slug(self):
        artifact = Artifact(
            job_id="Code Review!",
            job_name="Code Review",
            job_type="code_review",
            model="llama3:8b",
            priority="high",
            text="result",
            prompt="review this",
            iteration=0,
            tokens_generated=100,
            tokens_per_second=50.0,
            elapsed_seconds=2.0,
            timestamp="2026-08-04T12:00:00",
        )
        slug = artifact.slug()
        assert "2026-08-04" in slug
        assert "code-review" in slug
        assert "000" in slug

    def test_to_dict(self):
        artifact = Artifact(
            job_id="test",
            job_name="Test",
            job_type="custom",
            model="m",
            priority="low",
            text="output",
            prompt="input",
            iteration=1,
            tokens_generated=50,
            tokens_per_second=25.0,
            elapsed_seconds=2.0,
        )
        d = artifact.to_dict()
        assert d["job_id"] == "test"
        assert d["tokens_generated"] == 50
        assert "timestamp" in d

    def test_save(self, tmp_path):
        artifact = Artifact(
            job_id="test",
            job_name="Test",
            job_type="custom",
            model="m",
            priority="low",
            text="# Hello\n\nWorld",
            prompt="test",
            iteration=0,
            tokens_generated=10,
            tokens_per_second=5.0,
            elapsed_seconds=2.0,
        )
        md_path = artifact.save(tmp_path)
        assert md_path.exists()
        assert md_path.suffix == ".md"
        assert "Hello" in md_path.read_text()

        json_path = md_path.with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["job_id"] == "test"


class TestForgeStats:
    def test_to_dict(self):
        stats = ForgeStats()
        stats.iterations = 5
        stats.artifacts_produced = 3
        stats.total_tokens = 500
        stats.models_used.add("llama3:8b")
        d = stats.to_dict()
        assert d["iterations"] == 5
        assert d["artifacts_produced"] == 3
        assert d["total_tokens"] == 500
        assert "llama3:8b" in d["models_used"]

    def test_elapsed_seconds_no_start(self):
        stats = ForgeStats()
        assert stats.to_dict()["elapsed_seconds"] == 0.0


class TestForge:
    def test_dry_run(self, tmp_path):
        """Dry run should not call Ollama but should track iterations."""
        forge = Forge(output_dir=tmp_path / "output")
        session = ForgeSession(name="test", models=["test-model"])
        session.add_job(JobSpec(
            id="j1", name="Job 1", prompt="test", max_iterations=2
        ))

        stats = forge.run_session(
            session=session,
            max_iterations=2,
            dry_run=True,
        )
        assert stats.iterations == 2
        assert stats.artifacts_produced == 0  # dry run doesn't produce artifacts

    def test_state_management(self, tmp_path):
        forge = Forge(output_dir=tmp_path / "output")
        assert forge.state == ForgeState.IDLE

        state_data = forge._load_state()
        assert state_data is None  # nothing saved yet

    def test_get_status_empty(self, tmp_path):
        forge = Forge(output_dir=tmp_path / "output")
        status = forge.get_status()
        assert status["artifact_count"] == 0
        assert status["state"] in ("idle", "stopped")

    @patch("slackwater_forge.forge.OllamaClient")
    def test_ollama_not_available(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = False
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="j1", name="J1", prompt="x"))

        stats = forge.run_session(session=session)
        assert stats.iterations == 0
        assert forge.state == ForgeState.ERROR
