"""Tests for the Ollama model interface."""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from slackwater_forge.models import OllamaClient, ModelInfo, GenerateResult


class TestOllamaClient:
    @patch("httpx.Client")
    def test_is_available_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(status_code=200)
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        assert client.is_available() is True

    @patch("httpx.Client")
    def test_is_available_connection_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        assert client.is_available() is False

    @patch("httpx.Client")
    def test_list_models(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {
                        "name": "llama3:8b",
                        "size": 4661212032,
                        "details": {
                            "family": "llama",
                            "parameter_size": "8B",
                            "quantization_level": "Q4_0",
                        },
                    },
                    {
                        "name": "qwen2.5:0.5b",
                        "size": 395673888,
                        "details": {
                            "family": "qwen2",
                            "parameter_size": "0.5B",
                            "quantization_level": "Q4_0",
                        },
                    },
                ]
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        models = client.list_models()
        assert len(models) == 2
        assert models[0].name == "llama3:8b"
        assert models[0].family == "llama"
        assert models[1].parameter_size == "0.5B"

    @patch("httpx.Client")
    def test_generate(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": "Hello, world!",
                "model": "test-model",
                "eval_count": 10,
                "eval_duration": 1_000_000_000,
                "total_duration": 1_500_000_000,
                "created_at": "2026-08-04T12:00:00Z",
            },
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient()
        result = client.generate(model="test-model", prompt="Hello")

        assert result.text == "Hello, world!"
        assert result.model == "test-model"
        assert result.eval_count == 10
        assert result.tokens_per_second == 10.0
        assert result.elapsed_seconds == 1.5


class TestModelInfo:
    def test_from_api(self):
        raw = {
            "name": "test-model",
            "size": 1000000000,
            "details": {
                "family": "test",
                "parameter_size": "7B",
                "quantization_level": "Q4_0",
            },
        }
        info = ModelInfo.from_api(raw)
        assert info.name == "test-model"
        assert info.family == "test"
        assert info.parameter_size == "7B"
        assert info.quantization == "Q4_0"

    def test_from_api_defaults(self):
        info = ModelInfo.from_api({})
        assert info.name == "unknown"


class TestGenerateResult:
    def test_properties(self):
        result = GenerateResult(
            text="test",
            model="m",
            eval_count=100,
            eval_duration_ns=2_000_000_000,
            total_duration_ns=3_000_000_000,
        )
        assert result.elapsed_seconds == 3.0
        assert result.eval_seconds == 2.0

    def test_properties_zero(self):
        result = GenerateResult(text="t", model="m")
        assert result.elapsed_seconds == 0.0
        assert result.eval_seconds == 0.0
