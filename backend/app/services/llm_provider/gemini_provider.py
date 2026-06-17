"""
Gemini / Google AI Studio implementation of LLMProvider.

Uses the current official `google-genai` SDK (`from google import genai`;
`genai.Client(...).models.generate_content(...)`). The older
`google-generativeai` package (genai.configure + GenerativeModel) is
deprecated and is NOT used here.

Model IDs follow Google's naming (e.g. "gemini-2.5-flash",
"gemini-embedding-001") and are not normalized. If you specify an
Ollama-style tag like "gemma4:e2b" against this provider, the API will
return a 404 — that's a config error, not a runtime bug.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Gemini's free/shared tier intermittently returns transient errors
# (500 INTERNAL / 503 UNAVAILABLE / 429 RESOURCE_EXHAUSTED). These are not our
# bug and usually succeed on a quick retry, so wrap generate calls with backoff.
_RETRYABLE = ("500", "503", "429", "INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")


def _is_retryable(exc: Exception) -> bool:
    s = str(exc)
    return any(tok in s for tok in _RETRYABLE)


def _with_retry(fn, *, attempts: int = 3, base_delay: float = 1.5):
    """Call fn(); retry on transient Gemini errors with linear backoff."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i == attempts - 1 or not _is_retryable(e):
                raise
            delay = base_delay * (i + 1)
            logger.warning("Gemini transient error (attempt %d/%d), retrying in %.1fs: %s",
                           i + 1, attempts, delay, str(e)[:160])
            time.sleep(delay)
    if last:
        raise last


def _lazy_import():
    """Import the google-genai SDK only when this provider is actually used."""
    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
        return genai, genai_types
    except ImportError as e:
        raise RuntimeError(
            "google-genai not installed — run `uv add google-genai` "
            "(note: the package is `google-genai`, not the deprecated "
            "`google-generativeai`), or switch LLM_PROVIDER back to 'ollama'."
        ) from e


def _split_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split out the system message; return (system_instruction, role/content turns)."""
    system: Optional[str] = None
    turns: List[Dict[str, Any]] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            system = (system + "\n\n" + content) if system else content
            continue
        turns.append({"role": role, "content": content})
    return system, turns


def _is_gemma(model: Optional[str]) -> bool:
    """Gemma models on the Gemini API reject `system_instruction` (HTTP 400).

    For them the system prompt must be folded into the user content instead.
    """
    return "gemma" in (model or "").lower()


def _fold_system_for_gemma(
    system: Optional[str], turns: List[Dict[str, Any]]
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Prepend the system text to the first user turn; clear system_instruction."""
    if not system or not turns:
        return system, turns
    folded = [dict(t) for t in turns]
    for t in folded:
        if (t.get("role") or "").lower() == "user":
            t["content"] = f"{system}\n\n{t.get('content') or ''}"
            return None, folded
    # no user turn — drop system into a fresh leading user turn
    return None, [{"role": "user", "content": system}] + folded


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "gemini-2.5-flash",
        embed_model: str = "text-embedding-004",
        timeout: int = 120,
    ):
        if not api_key:
            raise ValueError("Gemini provider requires GEMINI_API_KEY")
        self._api_key = api_key
        self._default_model = default_model
        self._embed_model = embed_model
        self._timeout = timeout
        genai, genai_types = _lazy_import()
        self._types = genai_types
        # http_options.timeout is in milliseconds in the google-genai SDK
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout * 1000),
        )

    # ---- internal helpers ----------------------------------------------------

    def _build_config(
        self,
        *,
        system_instruction: Optional[str],
        format: Optional[Any],
        options: Optional[Dict[str, Any]],
    ):
        kwargs: Dict[str, Any] = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if format == "json":
            kwargs["response_mime_type"] = "application/json"
        if options:
            for k_src, k_dst in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("top_k", "top_k"),
                ("num_predict", "max_output_tokens"),
                ("max_output_tokens", "max_output_tokens"),
            ):
                val = options.get(k_src)
                if val is not None:
                    kwargs[k_dst] = val
        return self._types.GenerateContentConfig(**kwargs) if kwargs else None

    def _turns_to_contents(self, turns: List[Dict[str, Any]]) -> List[Any]:
        """Convert role/content turns to a list of genai Content objects.

        Gemini roles are 'user' and 'model'; map assistant -> model.
        """
        contents: List[Any] = []
        for turn in turns:
            role = turn.get("role") or "user"
            gem_role = "user" if role == "user" else "model"
            text = turn.get("content") or ""
            contents.append(
                self._types.Content(
                    role=gem_role,
                    parts=[self._types.Part.from_text(text=text)],
                )
            )
        return contents

    # ---- LLMProvider interface ----------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        format: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        target = model or self._default_model
        system, turns = _split_messages(messages)
        if _is_gemma(target):
            system, turns = _fold_system_for_gemma(system, turns)
        contents = self._turns_to_contents(turns) or [
            self._types.Content(role="user", parts=[self._types.Part.from_text(text="")])
        ]
        config = self._build_config(system_instruction=system, format=format, options=options)
        try:
            resp = _with_retry(lambda: self._client.models.generate_content(
                model=target,
                contents=contents,
                config=config,
            ))
        except Exception as e:
            logger.error("Gemini chat error: %s", e)
            raise RuntimeError(f"Gemini API 錯誤：{e}") from e
        return (resp.text or "").strip()

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, str]]:
        target = model or self._default_model
        system, turns = _split_messages(messages)
        if _is_gemma(target):
            system, turns = _fold_system_for_gemma(system, turns)
        contents = self._turns_to_contents(turns) or [
            self._types.Content(role="user", parts=[self._types.Part.from_text(text="")])
        ]
        config = self._build_config(system_instruction=system, format=None, options=options)
        try:
            stream = self._client.models.generate_content_stream(
                model=target,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield {"type": "content", "text": text}
        except Exception as e:
            logger.error("Gemini stream error: %s", e)
            raise RuntimeError(f"Gemini stream 錯誤：{e}") from e

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
        target = model or self._default_model
        if _is_gemma(target) and system:
            prompt = f"{system}\n\n{prompt or ''}"
            system = None
        parts: List[Any] = []
        if prompt:
            parts.append(self._types.Part.from_text(text=prompt))
        if images:
            # images expected as base64 strings (same shape as the Ollama API)
            import base64
            for b64 in images:
                try:
                    raw = base64.b64decode(b64)
                    parts.append(self._types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
                except Exception:
                    continue
        if not parts:
            parts = [self._types.Part.from_text(text="")]

        config = self._build_config(system_instruction=system, format=format, options=options)
        try:
            resp = _with_retry(lambda: self._client.models.generate_content(
                model=target,
                contents=[self._types.Content(role="user", parts=parts)],
                config=config,
            ))
        except Exception as e:
            logger.error("Gemini generate error: %s", e)
            raise RuntimeError(f"Gemini API 錯誤：{e}") from e
        return (resp.text or "").strip()

    def embed(
        self,
        inputs: List[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        target = model or self._embed_model
        cfg = self._types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        out: List[List[float]] = []
        for text in inputs:
            try:
                resp = self._client.models.embed_content(
                    model=target,
                    contents=(text or "")[:8000],
                    config=cfg,
                )
            except Exception as e:
                logger.error("Gemini embed error: %s", e)
                raise RuntimeError(f"Gemini embed 錯誤：{e}") from e
            # resp.embeddings is a list of ContentEmbedding; one per input string
            embeddings = getattr(resp, "embeddings", None) or []
            if not embeddings:
                raise RuntimeError("Gemini embed 回傳空向量")
            out.append(list(embeddings[0].values))
        return out

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            return [{"name": m.name} for m in self._client.models.list()]
        except Exception as e:
            logger.warning("Gemini list_models failed: %s", e)
            return []

    def version(self) -> Optional[str]:
        return "gemini-api"
