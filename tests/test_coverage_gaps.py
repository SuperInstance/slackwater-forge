"""Coverage gap tests for slackwater_forge — targeting forge.py (70%), models.py (74%), briefer.py (77%), jobs.py (96%).

Key gaps:
- forge.py: _format_prompt, _execute_job, run_session with real execution, get_status with saved state
- models.py: chat(), get_model(), pull_model(), context manager
- briefer.py: _ai_synthesize(), load_artifacts with markdown, generate with AI summary
- jobs.py: JobManager methods
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slackwater_forge.forge import Artifact, Forge, ForgeState, ForgeStats
from slackwater_forge.jobs import (
    BUILTIN_TEMPLATES,
    ForgeSession,
    JobManager,
    JobSpec,
    JobType,
    Priority,
)
from slackwater_forge.models import GenerateResult, ModelInfo, OllamaClient
from slackwater_forge.briefer import Briefer, _compute_confidence, _extract_preview, _categorize


# ── forge.py gaps ──────────────────────────────────────────────


class TestForgeFormatPrompt:
    def test_format_prompt_with_iteration(self):
        forge = Forge(output_dir="/tmp/forge-test")
        job = JobSpec(id="j1", name="Test", prompt="Iteration {iteration} of work")
        result = forge._format_prompt(job, 5)
        assert result == "Iteration 5 of work"

    def test_format_prompt_without_iteration(self):
        forge = Forge(output_dir="/tmp/forge-test")
        job = JobSpec(id="j1", name="Test", prompt="Static prompt")
        result = forge._format_prompt(job, 0)
        assert result == "Static prompt"


class TestForgeExecuteJob:
    @patch("slackwater_forge.forge.OllamaClient")
    def test_execute_job_success(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.generate.return_value = GenerateResult(
            text="Generated output",
            model="test-model",
            eval_count=50,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=1_500_000_000,
            tokens_per_second=50.0,
        )
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        job = JobSpec(id="j1", name="Job 1", prompt="test prompt", model="test-model")
        artifact = forge._execute_job(job, 0)

        assert artifact.job_id == "j1"
        assert artifact.text == "Generated output"
        assert artifact.iteration == 0
        assert artifact.tokens_generated == 50
        assert forge.stats.artifacts_produced == 1
        assert forge.stats.total_tokens == 50
        assert "test-model" in forge.stats.models_used

    @patch("slackwater_forge.forge.OllamaClient")
    def test_execute_job_calls_callback(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.generate.return_value = GenerateResult(
            text="ok", model="m", eval_count=1,
        )
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        job = JobSpec(id="j1", name="Job 1", prompt="x")

        callback_called = []
        def cb(art):
            callback_called.append(art)

        forge._execute_job(job, 0)  # no callback set
        # Now set callback and run again
        forge._on_iteration = cb
        forge._execute_job(job, 1)
        assert len(callback_called) == 1


class TestForgeRunSession:
    @patch("slackwater_forge.forge.OllamaClient")
    def test_no_enabled_jobs(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        session = ForgeSession(name="empty")
        stats = forge.run_session(session=session)
        assert stats.iterations == 0

    @patch("slackwater_forge.forge.OllamaClient")
    def test_max_iterations_limit(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = GenerateResult(
            text="out", model="m", eval_count=5,
        )
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="j1", name="J1", prompt="x", max_iterations=10))
        stats = forge.run_session(session=session, max_iterations=3)
        assert stats.iterations == 3

    @patch("slackwater_forge.forge.OllamaClient")
    def test_job_error_handling(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = RuntimeError("model failed")
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="j1", name="J1", prompt="x"))
        stats = forge.run_session(session=session, max_iterations=2)
        assert stats.errors == 2
        assert stats.iterations == 2

    @patch("slackwater_forge.forge.OllamaClient")
    def test_request_stop(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        forge.request_stop()
        assert forge._stop_requested is True
        assert forge.state == ForgeState.STOPPED

    @patch("slackwater_forge.forge.OllamaClient")
    def test_get_status_with_saved_state(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        # Save some state
        forge.state = ForgeState.RUNNING
        forge.stats.iterations = 5
        forge._save_state()

        status = forge.get_status()
        assert status["state"] == "running"
        assert status["stats"]["iterations"] == 5

    @patch("slackwater_forge.forge.OllamaClient")
    def test_load_state_corrupt_file(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        forge.state_file.write_text("{corrupt json")
        result = forge._load_state()
        assert result is None

    @patch("slackwater_forge.forge.OllamaClient")
    def test_iter_artifacts_with_corrupt_json(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        (tmp_path / "output" / "good.json").write_text(json.dumps({"job_id": "ok"}))
        (tmp_path / "output" / "bad.json").write_text("{broken")
        artifacts = forge._iter_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0]["job_id"] == "ok"

    @patch("slackwater_forge.forge.OllamaClient")
    def test_save_state_and_load(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        forge = Forge(output_dir=tmp_path / "output")
        forge.stats.iterations = 42
        forge.state = ForgeState.STOPPED
        forge._save_state()

        loaded = forge._load_state()
        assert loaded is not None
        assert loaded["state"] == "stopped"
        assert loaded["stats"]["iterations"] == 42


class TestArtifactEdgeCases:
    def test_slug_special_chars(self):
        art = Artifact(
            job_id="Job With Spaces & Specials!",
            job_name="Test",
            job_type="custom",
            model="m", priority="low", text="t", prompt="p",
            iteration=0, tokens_generated=1,
            tokens_per_second=1.0, elapsed_seconds=1.0,
        )
        slug = art.slug()
        assert " " not in slug
        assert "&" not in slug

    def test_to_dict_includes_verified_and_confidence(self):
        art = Artifact(
            job_id="x", job_name="X", job_type="custom", model="m",
            priority="low", text="t", prompt="p", iteration=0,
            tokens_generated=1, tokens_per_second=1.0, elapsed_seconds=1.0,
            verified=True, confidence=0.95,
        )
        d = art.to_dict()
        assert d["verified"] is True
        assert d["confidence"] == 0.95


class TestForgeStatsEdgeCases:
    def test_elapsed_with_invalid_start(self):
        stats = ForgeStats(start_time="not-a-date")
        assert stats._elapsed_seconds() == 0.0

    def test_elapsed_with_valid_start(self):
        import datetime
        stats = ForgeStats(start_time=datetime.datetime.now().isoformat())
        elapsed = stats._elapsed_seconds()
        assert elapsed >= 0.0


# ── models.py gaps ─────────────────────────────────────────────


class TestOllamaClientChat:
    @patch("httpx.Client")
    def test_chat_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "message": {"content": "Chat response!"},
                "model": "test-model",
                "eval_count": 20,
                "eval_duration": 2_000_000_000,
                "total_duration": 3_000_000_000,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        result = client.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result.text == "Chat response!"
        assert result.model == "test-model"
        assert result.eval_count == 20
        assert result.tokens_per_second == 10.0

    @patch("httpx.Client")
    def test_chat_with_options(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "message": {"content": "ok"},
                "model": "m",
                "eval_count": 5,
                "eval_duration": 500_000_000,
                "total_duration": 600_000_000,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        client.chat(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.5},
        )
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["options"]["temperature"] == 0.5


class TestOllamaClientGetModel:
    @patch("httpx.Client")
    def test_get_model_found(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {"name": "test-model:latest", "details": {}},
                    {"name": "other-model", "details": {}},
                ]
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        model = client.get_model("test-model")
        assert model is not None
        assert model.name == "test-model:latest"

    @patch("httpx.Client")
    def test_get_model_not_found(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": []},
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        model = client.get_model("nonexistent")
        assert model is None


class TestOllamaClientPullModel:
    @patch("httpx.Client")
    def test_pull_model_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        assert client.pull_model("test-model") is True

    @patch("httpx.Client")
    def test_pull_model_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(status_code=404)
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        assert client.pull_model("nonexistent") is False


class TestOllamaClientContextManager:
    def test_context_manager(self):
        client = OllamaClient()
        # Client starts as None
        assert client._client is None
        # Enter returns self
        with client as c:
            assert c is client
            # Accessing .client creates the httpx.Client
            _ = client.client
            assert client._client is not None
        # After exit, client is closed
        assert client._client is None

    def test_close_without_client(self):
        client = OllamaClient()
        client.close()  # should not raise


class TestGenerateWithSystemAndContext:
    @patch("httpx.Client")
    def test_generate_with_system_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": "ok", "model": "m",
                "eval_count": 1, "eval_duration": 0,
                "total_duration": 0,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        client.generate(model="m", prompt="hi", system="You are helpful.")
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["system"] == "You are helpful."

    @patch("httpx.Client")
    def test_generate_with_context(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": "ok", "model": "m",
                "eval_count": 1, "eval_duration": 0,
                "total_duration": 0,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        client.generate(model="m", prompt="hi", context=[1, 2, 3])
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["context"] == [1, 2, 3]

    @patch("httpx.Client")
    def test_generate_with_options(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": "ok", "model": "m",
                "eval_count": 1, "eval_duration": 0,
                "total_duration": 0,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        client.generate(model="m", prompt="hi", options={"temperature": 0.3})
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["options"]["temperature"] == 0.3


class TestModelInfoEdgeCases:
    def test_from_api_partial_details(self):
        info = ModelInfo.from_api({
            "name": "model-x",
            "details": {"family": "test"},
        })
        assert info.name == "model-x"
        assert info.family == "test"
        assert info.quantization == ""
        assert info.parameter_size == ""


# ── jobs.py gaps ───────────────────────────────────────────────


class TestJobManager:
    def test_create_and_load_session(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        session = mgr.create_session("test", description="test session")
        assert session.name == "test"

        loaded = mgr.load_session("test")
        assert loaded.name == "test"

    def test_list_sessions(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session("s1")
        mgr.create_session("s2")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_archive_session(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session("to_archive")
        result = mgr.archive_session("to_archive")
        assert result is not None
        assert result.exists()
        # Should no longer be in active
        assert mgr.load_session("to_archive") if False else True  # load would fail
        with pytest.raises(FileNotFoundError):
            mgr.load_session("to_archive")

    def test_archive_nonexistent(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        result = mgr.archive_session("nonexistent")
        assert result is None

    def test_delete_session(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        mgr.create_session("to_delete")
        assert mgr.delete_session("to_delete") is True
        assert mgr.delete_session("to_delete") is False  # already gone

    def test_save_session_with_name(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        session = ForgeSession(name="original")
        path = mgr.save_session(session, name="renamed")
        assert "renamed" in path.name
        assert session.name == "renamed"

    def test_create_job(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        job = mgr.create_job(
            job_id="test",
            name="Test Job",
            prompt="test prompt",
            job_type=JobType.CODE_REVIEW,
            priority=Priority.HIGH,
        )
        assert job.id == "test"
        assert job.type == JobType.CODE_REVIEW
        assert job.priority == Priority.HIGH

    def test_get_job_not_found(self):
        session = ForgeSession(name="test")
        assert session.get_job("nonexistent") is None

    def test_get_job_found(self):
        session = ForgeSession(name="test")
        job = JobSpec(id="j1", name="J1", prompt="x")
        session.add_job(job)
        found = session.get_job("j1")
        assert found is not None
        assert found.name == "J1"

    def test_remove_job(self):
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="j1", name="J1", prompt="x"))
        assert session.remove_job("j1") is True
        assert len(session.jobs) == 0

    def test_remove_job_not_found(self):
        session = ForgeSession(name="test")
        assert session.remove_job("nonexistent") is False

    def test_builtin_templates(self):
        assert "code-review" in BUILTIN_TEMPLATES
        assert "creative-writing" in BUILTIN_TEMPLATES
        assert "research" in BUILTIN_TEMPLATES
        assert "brainstorm" in BUILTIN_TEMPLATES
        assert "documentation" in BUILTIN_TEMPLATES


class TestForgeSessionEdgeCases:
    def test_from_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ForgeSession.from_file(tmp_path / "nonexistent.json")

    def test_disabled_jobs_filtered(self):
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="j1", name="J1", prompt="x", enabled=True))
        session.add_job(JobSpec(id="j2", name="J2", prompt="y", enabled=False))
        enabled = session.enabled_jobs()
        assert len(enabled) == 1
        assert enabled[0].id == "j1"


# ── briefer.py gaps ────────────────────────────────────────────


class TestBrieferAI:
    @patch("slackwater_forge.briefer.OllamaClient")
    def test_ai_synthesize_success(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        ai_result = GenerateResult(
            text="""## Executive Summary
Good progress overnight. All jobs completed successfully.

## Recommendations
1. Increase model context window
2. Review code quality findings
3. Expand test coverage""",
            model="m",
        )
        mock_client.generate.return_value = ai_result
        mock_client_cls.return_value = mock_client

        briefer = Briefer(output_dir=tmp_path)
        # Create artifact
        data = {
            "job_id": "test", "job_name": "Test", "job_type": "code_review",
            "model": "m", "priority": "high", "text": "Found 2 issues.",
            "prompt": "review", "iteration": 0, "tokens_generated": 10,
            "tokens_per_second": 5.0, "elapsed_seconds": 2.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefing = briefer.generate(model="m", use_ai_summary=True)
        assert briefing["summary"]  # should have AI summary
        assert len(briefing["recommendations"]) == 3

    @patch("slackwater_forge.briefer.OllamaClient")
    def test_ai_synthesize_fallback_on_error(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = RuntimeError("ollama down")
        mock_client_cls.return_value = mock_client

        briefer = Briefer(output_dir=tmp_path)
        data = {
            "job_id": "test", "job_name": "Test", "job_type": "analysis",
            "model": "m", "priority": "medium", "text": "Analysis output.",
            "prompt": "analyze", "iteration": 0, "tokens_generated": 5,
            "tokens_per_second": 5.0, "elapsed_seconds": 1.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefing = briefer.generate(model="m", use_ai_summary=True)
        # Should fall back to offline summary
        assert briefing["summary"]
        assert briefing["recommendations"] == []

    @patch("slackwater_forge.briefer.OllamaClient")
    def test_ai_synthesize_no_recommendations_section(self, mock_client_cls, tmp_path):
        """AI returns text without recommendations section."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = GenerateResult(
            text="""## Executive Summary
Just a summary without recommendations.""",
            model="m",
        )
        mock_client_cls.return_value = mock_client

        briefer = Briefer(output_dir=tmp_path)
        data = {
            "job_id": "test", "job_name": "Test", "job_type": "custom",
            "model": "m", "priority": "low", "text": "output",
            "prompt": "p", "iteration": 0, "tokens_generated": 5,
            "tokens_per_second": 5.0, "elapsed_seconds": 1.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefing = briefer.generate(model="m", use_ai_summary=True)
        assert briefing["summary"]
        assert briefing["recommendations"] == []

    @patch("slackwater_forge.briefer.OllamaClient")
    def test_ai_synthesize_no_summary_header(self, mock_client_cls, tmp_path):
        """AI returns text without standard headers."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = GenerateResult(
            text="Just some plain text response without headers.",
            model="m",
        )
        mock_client_cls.return_value = mock_client

        briefer = Briefer(output_dir=tmp_path)
        data = {
            "job_id": "test", "job_name": "Test", "job_type": "custom",
            "model": "m", "priority": "low", "text": "output",
            "prompt": "p", "iteration": 0, "tokens_generated": 5,
            "tokens_per_second": 5.0, "elapsed_seconds": 1.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefing = briefer.generate(model="m", use_ai_summary=True)
        assert briefing["summary"]  # Falls back to full text


class TestBrieferLoadArtifacts:
    def test_load_artifacts_with_md_missing(self, tmp_path):
        """When .md file is missing, text comes from json data."""
        data = {"job_id": "test", "job_name": "T", "text": "from json"}
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))

        briefer = Briefer(output_dir=tmp_path)
        artifacts = briefer.load_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0]["full_text"] == "from json"

    def test_load_artifacts_ignores_hidden(self, tmp_path):
        (tmp_path / ".hidden.json").write_text(json.dumps({"job_id": "x"}))
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps({"job_id": "y", "text": "hello"}))

        briefer = Briefer(output_dir=tmp_path)
        artifacts = briefer.load_artifacts()
        assert len(artifacts) == 1

    def test_load_artifacts_ignores_corrupt_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{broken")
        (tmp_path / "2026-08-04-good-000.json").write_text(json.dumps({"job_id": "y", "text": "ok"}))

        briefer = Briefer(output_dir=tmp_path)
        artifacts = briefer.load_artifacts()
        assert len(artifacts) == 1


class TestBrieferSaveFormats:
    def test_save_md_only(self, tmp_path):
        data = {
            "job_id": "t", "job_name": "T", "job_type": "custom",
            "model": "m", "priority": "low", "text": "x",
            "prompt": "p", "iteration": 0, "tokens_generated": 1,
            "tokens_per_second": 1.0, "elapsed_seconds": 1.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-t-000.json").write_text(json.dumps(data))
        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)
        paths = briefer.save_briefing(briefing, formats=["md"])
        assert len(paths) == 1
        assert paths[0].suffix == ".md"

    def test_save_to_custom_dir(self, tmp_path):
        data = {
            "job_id": "t", "job_name": "T", "job_type": "custom",
            "model": "m", "priority": "low", "text": "x",
            "prompt": "p", "iteration": 0, "tokens_generated": 1,
            "tokens_per_second": 1.0, "elapsed_seconds": 1.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-t-000.json").write_text(json.dumps(data))
        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)
        custom_dir = tmp_path / "reports"
        paths = briefer.save_briefing(briefing, output_dir=custom_dir, formats=["md", "html"])
        assert len(paths) == 2
        assert custom_dir.exists()


class TestHelperEdgeCases:
    def test_extract_preview_markdown_headers(self):
        text = "## Header\n\nSome content."
        preview = _extract_preview(text)
        assert "##" not in preview  # Headers are stripped

    def test_extract_preview_multiline(self):
        text = "Line 1\n\nLine 2\n\nLine 3"
        preview = _extract_preview(text)
        assert "Line 1" in preview
        assert "Line 2" in preview

    def test_compute_confidence_with_verified(self):
        artifacts = [{"job_type": "x", "verified": True}] * 5
        conf = _compute_confidence(artifacts)
        assert conf > 0.5  # verified bonus

    def test_compute_confidence_single_type(self):
        artifacts = [{"job_type": "same"}] * 10
        conf = _compute_confidence(artifacts)
        assert 0 < conf <= 1.0

    def test_categorize_missing_type(self):
        artifacts = [{"job_type": None}]
        cats = _categorize(artifacts)
        assert None in cats

    def test_categorize_explicit_other(self):
        artifacts = [{"job_type": "custom"}]
        cats = _categorize(artifacts)
        assert "custom" in cats
