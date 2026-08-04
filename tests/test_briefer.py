"""Tests for the briefer module."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slackwater_forge.briefer import Briefer, _extract_preview, _categorize, _compute_confidence


class TestHelpers:
    def test_extract_preview(self):
        text = "# Title\n\nSome **bold** text.\n\nMore content here."
        preview = _extract_preview(text)
        assert "Title" in preview
        assert len(preview) <= 305  # max_chars + ellipsis

    def test_extract_preview_empty(self):
        assert _extract_preview("") == ""

    def test_extract_preview_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        preview = _extract_preview(text)
        assert "[code block]" in preview
        assert "print" not in preview

    def test_categorize(self):
        artifacts = [
            {"job_type": "code_review", "text": "a"},
            {"job_type": "code_review", "text": "b"},
            {"job_type": "research", "text": "c"},
        ]
        cats = _categorize(artifacts)
        assert len(cats["code_review"]) == 2
        assert len(cats["research"]) == 1

    def test_compute_confidence_empty(self):
        assert _compute_confidence([]) == 0.0

    def test_compute_confidence_many(self):
        artifacts = [{"job_type": "code_review"}, {"job_type": "research"}]
        conf = _compute_confidence(artifacts)
        assert 0 < conf <= 1.0

    def test_compute_confidence_diverse(self):
        artifacts = [
            {"job_type": t, "verified": True}
            for t in ["code_review", "research", "creative_writing", "analysis"]
        ]
        conf = _compute_confidence(artifacts)
        assert conf > 0.5  # diverse + verified = higher confidence


class TestBriefer:
    def test_load_artifacts_empty(self, tmp_path):
        briefer = Briefer(output_dir=tmp_path)
        artifacts = briefer.load_artifacts()
        assert artifacts == []

    def test_load_artifacts(self, tmp_path):
        # Create some fake artifacts
        for i in range(3):
            data = {
                "job_id": f"job_{i}",
                "job_name": f"Job {i}",
                "job_type": "code_review",
                "model": "test-model",
                "priority": "high" if i == 0 else "low",
                "text": f"Output {i}" * 20,
                "prompt": "test",
                "iteration": i,
                "tokens_generated": 100 * (i + 1),
                "tokens_per_second": 50.0,
                "elapsed_seconds": 2.0,
                "timestamp": f"2026-08-04T12:0{i}:00",
            }
            slug = f"2026-08-04-job-{i}-{i:03d}"
            (tmp_path / f"{slug}.json").write_text(json.dumps(data))
            (tmp_path / f"{slug}.md").write_text(data["text"])

        briefer = Briefer(output_dir=tmp_path)
        artifacts = briefer.load_artifacts()
        assert len(artifacts) == 3
        assert artifacts[0]["job_id"] == "job_0"

    def test_generate_offline(self, tmp_path):
        # Create fake artifact
        data = {
            "job_id": "test",
            "job_name": "Test Job",
            "job_type": "analysis",
            "model": "test-model",
            "priority": "high",
            "text": "This is a test output about important findings.",
            "prompt": "analyze",
            "iteration": 0,
            "tokens_generated": 50,
            "tokens_per_second": 25.0,
            "elapsed_seconds": 2.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)

        assert briefing["date"]
        assert len(briefing["artifacts"]) == 1
        assert briefing["total_tokens"] == 50
        assert "test-model" in briefing["models"]
        assert briefing["confidence"] > 0
        assert briefing["summary"]  # Should have offline summary

    def test_to_markdown(self, tmp_path):
        data = {
            "job_id": "test",
            "job_name": "Test Job",
            "job_type": "code_review",
            "model": "test-model",
            "priority": "high",
            "text": "Found 3 issues.",
            "prompt": "review",
            "iteration": 0,
            "tokens_generated": 42,
            "tokens_per_second": 21.0,
            "elapsed_seconds": 2.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)
        md = briefer.to_markdown(briefing)

        assert "Morning Briefing" in md
        assert "Test Job" in md
        assert "42" in md  # token count

    def test_to_html(self, tmp_path):
        data = {
            "job_id": "test",
            "job_name": "Test Job",
            "job_type": "research",
            "model": "test-model",
            "priority": "medium",
            "text": "Research findings.",
            "prompt": "research",
            "iteration": 0,
            "tokens_generated": 100,
            "tokens_per_second": 50.0,
            "elapsed_seconds": 2.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)
        html = briefer.to_html(briefing)

        assert "<html" in html
        assert "Morning Briefing" in html
        assert "Test Job" in html

    def test_save_briefing(self, tmp_path):
        data = {
            "job_id": "test",
            "job_name": "Test",
            "job_type": "custom",
            "model": "m",
            "priority": "low",
            "text": "output",
            "prompt": "p",
            "iteration": 0,
            "tokens_generated": 10,
            "tokens_per_second": 5.0,
            "elapsed_seconds": 2.0,
            "timestamp": "2026-08-04T12:00:00",
        }
        (tmp_path / "2026-08-04-test-000.json").write_text(json.dumps(data))
        (tmp_path / "2026-08-04-test-000.md").write_text(data["text"])

        briefer = Briefer(output_dir=tmp_path)
        briefing = briefer.generate(use_ai_summary=False)
        paths = briefer.save_briefing(briefing, formats=["md", "html"])

        assert len(paths) == 2
        assert any(p.suffix == ".md" for p in paths)
        assert any(p.suffix == ".html" for p in paths)
