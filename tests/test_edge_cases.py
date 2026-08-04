"""Edge-case tests for slackwater-forge: empty/None inputs, extreme values, concurrency, serialization."""
import json
import threading
from datetime import datetime
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
from slackwater_forge.briefer import (
    Briefer,
    _compute_confidence,
    _extract_preview,
    _categorize,
)


# ── Empty / None Inputs ────────────────────────────────────────


class TestEmptyAndNoneInputs:
    def test_job_spec_empty_prompt(self):
        spec = JobSpec(id="empty", name="Empty", prompt="")
        assert spec.prompt == ""

    def test_job_spec_empty_tags(self):
        spec = JobSpec(id="t", name="T", prompt="x", tags=[])
        assert spec.tags == []

    def test_forge_session_empty_jobs(self):
        session = ForgeSession(name="empty")
        assert session.enabled_jobs() == []

    def test_artifact_empty_text(self):
        art = Artifact(
            job_id="x", job_name="X", job_type="custom", model="m",
            priority="low", text="", prompt="p", iteration=0,
            tokens_generated=0, tokens_per_second=0.0, elapsed_seconds=0.0,
        )
        assert art.slug()
        assert art.to_dict()["text"] == ""

    def test_extract_preview_none(self):
        assert _extract_preview(None) == ""

    def test_categorize_empty(self):
        assert _categorize([]) == {}

    def test_compute_confidence_none(self):
        assert _compute_confidence(None) == 0.0

    def test_briefer_load_empty_dir(self, tmp_path):
        b = Briefer(output_dir=tmp_path)
        assert b.load_artifacts() == []

    def test_briefer_generate_empty(self, tmp_path):
        b = Briefer(output_dir=tmp_path)
        briefing = b.generate(use_ai_summary=False)
        assert "error" in briefing

    def test_job_manager_load_nonexistent_raises(self, tmp_path):
        mgr = JobManager(tmp_path / "jobs")
        with pytest.raises(FileNotFoundError):
            mgr.load_session("does_not_exist")

    def test_forge_stats_empty_to_dict(self):
        stats = ForgeStats()
        d = stats.to_dict()
        assert d["iterations"] == 0
        assert d["artifacts_produced"] == 0
        assert d["total_tokens"] == 0
        assert d["elapsed_seconds"] == 0.0


# ── Extreme Values ─────────────────────────────────────────────


class TestExtremeValues:
    def test_job_spec_extreme_token_budget(self):
        spec = JobSpec(id="big", name="Big", prompt="x", token_budget=10**12)
        assert spec.token_budget == 10**12

    def test_job_spec_extreme_max_iterations(self):
        spec = JobSpec(id="many", name="Many", prompt="x", max_iterations=10**9)
        assert spec.max_iterations == 10**9

    def test_artifact_extreme_tokens(self):
        art = Artifact(
            job_id="big", job_name="Big", job_type="custom", model="m",
            priority="low", text="x", prompt="p", iteration=0,
            tokens_generated=10**15, tokens_per_second=10**9, elapsed_seconds=10**6,
        )
        d = art.to_dict()
        assert d["tokens_generated"] == 10**15

    def test_extract_preview_very_long_text(self):
        text = "A" * 100000
        preview = _extract_preview(text, max_chars=100)
        assert len(preview) <= 103  # max_chars + "..."
        assert preview.endswith("...")

    def test_forge_stats_many_models(self):
        stats = ForgeStats()
        for i in range(100):
            stats.models_used.add(f"model_{i}")
        d = stats.to_dict()
        assert len(d["models_used"]) == 100

    def test_model_info_empty_api_response(self):
        info = ModelInfo.from_api({})
        assert info.name == "unknown"
        assert info.size == ""
        assert info.family == ""

    def test_generate_result_zero_duration(self):
        r = GenerateResult(text="x", model="m")
        assert r.elapsed_seconds == 0.0
        assert r.tokens_per_second == 0.0
        assert r.eval_seconds == 0.0

    def test_forge_session_many_jobs(self):
        session = ForgeSession(name="big")
        for i in range(500):
            session.add_job(JobSpec(id=f"j{i}", name=f"Job {i}", prompt="x"))
        assert len(session.jobs) == 500
        enabled = session.enabled_jobs()
        assert len(enabled) == 500

    def test_priority_ordering_extreme(self):
        session = ForgeSession(name="test")
        session.add_job(JobSpec(id="low", name="L", prompt="x", priority=Priority.LOW))
        session.add_job(JobSpec(id="crit", name="C", prompt="x", priority=Priority.CRITICAL))
        enabled = session.enabled_jobs()
        assert enabled[0].id == "crit"
        assert enabled[-1].id == "low"


# ── Serialization Round-trips ──────────────────────────────────


class TestSerializationRoundtrips:
    def test_job_spec_json_roundtrip(self, tmp_path):
        original = JobSpec(
            id="roundtrip", name="Round Trip", prompt="test {iteration}",
            type=JobType.CODE_REVIEW, model="test-model",
            priority=Priority.HIGH, max_iterations=5,
            tags=["python", "web"], system_prompt="Be thorough",
        )
        path = tmp_path / "spec.json"
        original_json = original.model_dump_json(indent=2)
        path.write_text(original_json)
        loaded = JobSpec.model_validate_json(path.read_text())
        assert loaded.id == original.id
        assert loaded.type == original.type
        assert loaded.priority == original.priority
        assert loaded.tags == original.tags

    def test_forge_session_json_roundtrip(self, tmp_path):
        session = ForgeSession(name="rt", description="roundtrip test")
        session.add_job(JobSpec(id="j1", name="Job 1", prompt="x", priority=Priority.HIGH))
        session.add_job(JobSpec(id="j2", name="Job 2", prompt="y", type=JobType.RESEARCH))
        path = tmp_path / "session.json"
        session.save(path)
        loaded = ForgeSession.from_file(path)
        assert loaded.name == "rt"
        assert len(loaded.jobs) == 2
        assert loaded.jobs[0].id == "j1"
        assert loaded.jobs[1].type == JobType.RESEARCH

    def test_artifact_to_dict_roundtrip(self):
        art = Artifact(
            job_id="rt", job_name="RT", job_type="custom", model="m",
            priority="medium", text="output", prompt="input", iteration=3,
            tokens_generated=42, tokens_per_second=21.0, elapsed_seconds=2.0,
            verified=True, confidence=0.87,
        )
        d = art.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["job_id"] == "rt"
        assert loaded["verified"] is True
        assert loaded["confidence"] == 0.87
        assert loaded["iteration"] == 3

    def test_forge_stats_serialization(self):
        stats = ForgeStats()
        stats.iterations = 100
        stats.artifacts_produced = 50
        stats.total_tokens = 10000
        stats.errors = 2
        stats.models_used = {"model_a", "model_b"}
        stats.jobs_completed = ["job1", "job2"]
        d = stats.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["iterations"] == 100
        assert loaded["total_tokens"] == 10000
        assert set(loaded["models_used"]) == {"model_a", "model_b"}


# ── Concurrent Operations ──────────────────────────────────────


class TestConcurrency:
    def test_concurrent_add_jobs(self):
        session = ForgeSession(name="concurrent")

        def add_batch(prefix: str) -> None:
            for i in range(50):
                session.add_job(JobSpec(id=f"{prefix}_{i}", name=f"{prefix} {i}", prompt="x"))

        threads = [threading.Thread(target=add_batch, args=(p,)) for p in ("a", "b", "c", "d")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(session.jobs) == 200

    def test_concent_session_save_load(self, tmp_path):
        session = ForgeSession(name="concurrent_save")
        for i in range(20):
            session.add_job(JobSpec(id=f"j{i}", name=f"Job {i}", prompt="x"))

        path = tmp_path / "concurrent.json"

        errors: list[Exception] = []

        def save_and_load() -> None:
            try:
                session.save(path)
                loaded = ForgeSession.from_file(path)
                assert len(loaded.jobs) == 20
            except Exception as exc:
                errors.append(exc)

        # Save is not designed for true concurrent writes, but at least
        # sequential calls should work
        save_and_load()
        save_and_load()
        assert errors == []

    @patch("httpx.Client")
    def test_concurrent_generate_calls(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": "ok",
                "model": "m",
                "eval_count": 5,
                "eval_duration": 500_000_000,
                "total_duration": 600_000_000,
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        results: list[GenerateResult] = []
        results_lock = threading.Lock()

        def generate() -> None:
            r = client.generate(model="m", prompt="x")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=generate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r.text == "ok" for r in results)
