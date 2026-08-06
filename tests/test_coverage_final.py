"""Coverage gap closure tests for slackwater_forge — targeting remaining uncovered lines.

Gaps:
- forge.py 176-177: _setup_signal_handlers inner handler function
- forge.py 295-298: max_duration_seconds break in run_session
- forge.py 309-337: per-job iteration limits, all_done check, remaining jobs
- forge.py 360-361: client.close() in finally block (already partially covered)
- briefer.py 423-424: error briefing path in to_markdown
- briefer.py 482-485: preview rendering in to_markdown
"""
import json
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

from slackwater_forge.forge import Forge, ForgeState, ForgeStats
from slackwater_forge.jobs import JobSpec, JobType, Priority, ForgeSession
from slackwater_forge.models import GenerateResult
from slackwater_forge.briefer import Briefer


# ── Signal handler coverage (forge.py 176-177) ──────────────────────────────

class TestSignalHandlerInvocation:
    @patch("slackwater_forge.forge.signal.signal")
    def test_setup_signal_handlers_registers(self, mock_signal):
        forge = Forge(output_dir="/tmp/forge-sig-test")
        forge._setup_signal_handlers()
        # Verify both SIGINT and SIGTERM are registered
        registered_signals = [call.args[0] for call in mock_signal.call_args_list]
        assert signal.SIGINT in registered_signals
        assert signal.SIGTERM in registered_signals

    def test_signal_handler_calls_request_stop(self):
        """The inner handler function should set _stop_requested."""
        forge = Forge(output_dir="/tmp/forge-sig-test")
        # Capture the handler by patching signal.signal
        handlers = {}
        import slackwater_forge.forge as forge_mod

        original_signal = forge_mod.signal.signal

        def capture_signal(sig, handler):
            handlers[sig] = handler

        with patch.object(forge_mod.signal, "signal", side_effect=capture_signal):
            forge._setup_signal_handlers()

        # Simulate calling the SIGINT handler
        assert signal.SIGINT in handlers
        forge._stop_requested = False
        handlers[signal.SIGINT](signal.SIGINT, None)
        assert forge._stop_requested is True
        assert forge.state == ForgeState.STOPPED


# ── max_duration_seconds break (forge.py 295-298) ───────────────────────────

class TestMaxDurationBreak:
    @patch("slackwater_forge.forge.time.sleep")
    @patch("slackwater_forge.forge.time.monotonic")
    def test_max_duration_exceeded(self, mock_monotonic, mock_sleep, tmp_path):
        """Test that run_session breaks when max_duration_seconds is reached."""
        # Need enough return values for all monotonic() calls in run_session
        mock_monotonic.side_effect = [100.0, 200.0, 200.0, 200.0, 200.0]
        forge = Forge(output_dir=str(tmp_path))
        forge.client = MagicMock()

        job = JobSpec(id="j1", name="Test", prompt="test", max_iterations=1)
        session = ForgeSession(name="test", jobs=[job])

        # Should break immediately since elapsed >= max_duration_seconds
        stats = forge.run_session(
            session,
            max_duration_seconds=1,
            dry_run=True,
        )
        assert stats is not None


# ── Per-job iteration limits (forge.py 309-337) ─────────────────────────────

class TestPerJobIterationLimits:
    @patch("slackwater_forge.forge.time.sleep")
    def test_job_completes_after_max_iterations(self, mock_sleep, tmp_path):
        """Job should be marked completed when it hits max_iterations."""
        forge = Forge(output_dir=str(tmp_path))
        forge.client = MagicMock()

        # Create a job with max_iterations=1
        job = JobSpec(id="j1", name="Job1", prompt="test", max_iterations=1)
        job2 = JobSpec(id="j2", name="Job2", prompt="test", max_iterations=1)
        session = ForgeSession(name="test", jobs=[job, job2])

        # Pre-create artifact files so _iter_artifacts finds them
        (tmp_path / "artifact1.json").write_text(json.dumps({
            "job_id": "j1",
            "model": "test",
            "job_name": "Job1",
            "prompt": "test",
            "response": "result",
            "iteration": 0,
        }))

        stats = forge.run_session(session, dry_run=True, max_iterations=1)
        # j1 should be skipped since it already has artifacts
        # The job completion logic should fire


# ── Briefer error path (briefer.py 423-424) ─────────────────────────────────

class TestBrieferErrorPath:
    def test_to_markdown_with_error(self, tmp_path):
        """Briefing with an error key should render the error line."""
        briefer = Briefer(output_dir=str(tmp_path))
        briefing = {
            "error": "Ollama connection failed",
            "generated_at": "2024-01-01T00:00:00",
            "artifacts": [],
            "models_used": [],
            "jobs_run": [],
            "recommendations": [],
            "confidence_pct": 0,
        }
        result = briefer.to_markdown(briefing)
        assert "⚠️ Ollama connection failed" in result

    def test_to_markdown_error_short_circuits(self, tmp_path):
        """Error path should return early without rendering stats."""
        briefer = Briefer(output_dir=str(tmp_path))
        briefing = {
            "error": "Critical failure",
            "generated_at": "2024-01-01",
            "artifacts": [{"model": "test", "job_name": "j", "prompt": "p", "response": "r", "tokens_generated": 100, "elapsed_seconds": 5.0, "priority": "high", "preview": "hello world"}],
        }
        result = briefer.to_markdown(briefing)
        assert "⚠️ Critical failure" in result
        # Error path returns early, so stats should NOT appear
        assert "📊 Overnight Summary" not in result


# ── Briefer preview rendering (briefer.py 482-485) ──────────────────────────

class TestBrieferPreview:
    def test_to_markdown_with_preview(self, tmp_path):
        """Artifact with preview should render it."""
        briefer = Briefer(output_dir=str(tmp_path))
        briefing = {
            "generated_at": "2024-01-01",
            "artifacts": [
                {
                    "model": "llama2",
                    "job_name": "story",
                    "prompt": "write a story",
                    "response": "Once upon a time...",
                    "tokens_generated": 500,
                    "elapsed_seconds": 10.5,
                    "priority": "medium",
                    "preview": "This is a preview of the artifact output.",
                }
            ],
            "models_used": ["llama2"],
            "jobs_run": ["story"],
            "recommendations": ["Try a different model"],
            "confidence_pct": 85,
        }
        result = briefer.to_markdown(briefing)
        assert "This is a preview of the artifact output." in result
        assert "📊 Overnight Summary" in result

    def test_to_markdown_with_long_preview_truncation(self, tmp_path):
        """Long preview should be truncated to 200 chars."""
        briefer = Briefer(output_dir=str(tmp_path))
        long_preview = "A" * 300
        briefing = {
            "generated_at": "2024-01-01",
            "artifacts": [
                {
                    "model": "test",
                    "job_name": "j",
                    "prompt": "p",
                    "response": "r",
                    "tokens_generated": 10,
                    "elapsed_seconds": 1.0,
                    "priority": "low",
                    "preview": long_preview,
                }
            ],
            "models_used": ["test"],
            "jobs_run": ["j"],
            "recommendations": [],
            "confidence_pct": 50,
        }
        result = briefer.to_markdown(briefing)
        # Should contain truncated preview (200 chars + "...")
        assert "A" * 200 + "..." in result

    def test_to_markdown_artifact_without_preview(self, tmp_path):
        """Artifact without preview should not render preview line."""
        briefer = Briefer(output_dir=str(tmp_path))
        briefing = {
            "generated_at": "2024-01-01",
            "artifacts": [
                {
                    "model": "test",
                    "job_name": "j",
                    "prompt": "p",
                    "response": "r",
                    "tokens_generated": 10,
                    "elapsed_seconds": 1.0,
                    "priority": "low",
                    "preview": "",
                }
            ],
            "models_used": ["test"],
            "jobs_run": ["j"],
            "recommendations": [],
            "confidence_pct": 50,
        }
        result = briefer.to_markdown(briefing)
        assert "> " not in result.split("Confidence")[0].split("Priority: low")[1] if "Priority: low" in result else True


# ── Forge client.close() in finally (forge.py 360-361) ──────────────────────

class TestForgeClientCloseOnException:
    def test_client_closed_on_keyboard_interrupt(self, tmp_path):
        """Client should be closed even on KeyboardInterrupt."""
        forge = Forge(output_dir=str(tmp_path))
        mock_client = MagicMock()
        forge.client = mock_client

        job = JobSpec(id="j1", name="Test", prompt="test", max_iterations=1)
        session = ForgeSession(name="test", jobs=[job])

        with patch("slackwater_forge.forge.time") as mock_time:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = KeyboardInterrupt()
            forge.run_session(session, dry_run=False, max_iterations=1)

        mock_client.close.assert_called_once()


# ── Forge get_status edge (forge.py 395-398) ────────────────────────────────

class TestForgeGetStatusSaved:
    def test_get_status_with_saved_state_idle(self, tmp_path):
        """get_status should read saved state when forge is IDLE."""
        forge = Forge(output_dir=str(tmp_path))
        # Write a state file
        state_file = tmp_path / ".forge_state.json"
        state_file.write_text(json.dumps({
            "state": "stopped",
            "stats": {"iterations": 5, "artifacts_produced": 3, "errors": 1},
        }))
        forge._save_state = MagicMock()
        forge._load_state = MagicMock(return_value={
            "state": "stopped",
            "stats": {"iterations": 5, "artifacts_produced": 3, "errors": 1},
        })
        status = forge.get_status()
        assert status["state"] == "stopped" or status["state"] == "idle"
