"""Ollama implementation of LLMProvider — thin wrapper around OllamaClient."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..ollama_client import OllamaClient


class OllamaProvider:
    name = "ollama"

    def __init__(self, client: OllamaClient, default_model: Optional[str] = None, embed_model: Optional[str] = None):
        self._client = client
        self._default_model = default_model or client.default_model
        self._embed_model = embed_model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._client.chat(
            messages,
            model=model or self._default_model,
            format=format,
            options=options,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, str]]:
        yield from self._client.chat_stream(
            messages,
            model=model or self._default_model,
            options=options,
        )

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._client.generate(
            prompt,
            model=model or self._default_model,
            images=images,
            system=system,
            format=format,
            options=options,
        )

    def embed(
        self,
        inputs: List[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        return self._client.embed(inputs, model=model or self._embed_model)

    def list_models(self) -> List[Dict[str, Any]]:
        return self._client.list_models()

    def version(self) -> Optional[str]:
        return self._client.version()
