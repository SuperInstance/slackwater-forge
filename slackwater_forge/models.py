"""
Ollama model interface — handles all API communication with local Ollama instances.

Works with ANY Ollama model (Granite, Qwen, Llama, Mistral, etc.).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class ModelInfo:
    """Information about an available Ollama model."""
    name: str
    size: str = ""
    quantization: str = ""
    family: str = ""
    parameter_size: str = ""

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ModelInfo":
        return cls(
            name=raw.get("name", "unknown"),
            size=raw.get("size", ""),
            quantization=raw.get("details", {}).get("quantization_level", ""),
            family=raw.get("details", {}).get("family", ""),
            parameter_size=raw.get("details", {}).get("parameter_size", ""),
        )


@dataclass
class GenerateResult:
    """Result from a single Ollama generation call."""
    text: str
    model: str
    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    tokens_per_second: float = 0.0
    created_at: str = ""

    @property
    def elapsed_seconds(self) -> float:
        return self.total_duration_ns / 1e9 if self.total_duration_ns else 0.0

    @property
    def eval_seconds(self) -> float:
        return self.eval_duration_ns / 1e9 if self.eval_duration_ns else 0.0


class OllamaClient:
    """
    Synchronous + async-friendly client for Ollama's HTTP API.

    Supports both /api/generate and /api/chat endpoints.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[ModelInfo]:
        """List all available models in the Ollama instance."""
        resp = self.client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [ModelInfo.from_api(m) for m in data.get("models", [])]

    def get_model(self, name: str) -> ModelInfo | None:
        """Get info for a specific model."""
        for model in self.list_models():
            if model.name == name or model.name.startswith(name + ":"):
                return model
        return None

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        context: list[int] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> GenerateResult:
        """
        Generate text using /api/generate endpoint.

        Args:
            model: Ollama model name (e.g., "granite3.1-dense:2b")
            prompt: The prompt text
            system: Optional system prompt
            context: Optional context tokens from previous call
            stream: If True, returns an iterator (caller must handle)
            options: Additional Ollama options (temperature, num_ctx, etc.)
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options

        resp = self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tps = 0.0
        if eval_duration_ns > 0 and eval_count > 0:
            tps = eval_count / (eval_duration_ns / 1e9)

        return GenerateResult(
            text=data.get("response", ""),
            model=data.get("model", model),
            eval_count=eval_count,
            eval_duration_ns=eval_duration_ns,
            total_duration_ns=data.get("total_duration", 0),
            tokens_per_second=tps,
            created_at=data.get("created_at", ""),
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> GenerateResult:
        """Use the /api/chat endpoint with message history."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        resp = self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tps = 0.0
        if eval_duration_ns > 0 and eval_count > 0:
            tps = eval_count / (eval_duration_ns / 1e9)

        return GenerateResult(
            text=message.get("content", ""),
            model=data.get("model", model),
            eval_count=eval_count,
            eval_duration_ns=eval_duration_ns,
            total_duration_ns=data.get("total_duration", 0),
            tokens_per_second=tps,
            created_at=data.get("created_at", ""),
        )

    def pull_model(self, name: str) -> bool:
        """Pull a model from the Ollama registry."""
        resp = self.client.post(
            f"{self.base_url}/api/pull",
            json={"name": name},
            timeout=self.timeout,
        )
        return resp.status_code == 200

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
