"""
Provider-neutral LLM interface used by Agent, KG extractor, and any new RAG
code. Existing Ollama-specific call sites continue to use ollama_client.get_client()
directly so we can migrate incrementally.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Protocol


class LLMProvider(Protocol):
    name: str

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, str]]:
        ...

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
        ...

    def embed(
        self,
        inputs: List[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        ...

    def list_models(self) -> List[Dict[str, Any]]:
        ...

    def version(self) -> Optional[str]:
        ...
