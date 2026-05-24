"""Ollama client wrapper for Gemma 4 E4B (text + vision + audio).

Schema note — Ollama overloads the `images` field on a user message
to carry any binary media (images AND audio). This diverges from the
OpenAI Vision schema (typed content array) and is the single most
common porting bug. We normalize all media to base64 strings.

Best-practice ordering — media items go in the prompt before any
text, per Ollama's docs. The `images` list is preserved insertion-order.

Known issue (Ollama #15333, May 2026) — audio inference on
`gemma4:e4b` can intermittently crash the GGML forward pass. The
caller (`voice.py`) is responsible for retry-with-fallback.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import ollama

MODEL = "gemma4:e4b"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Outcome of a one-shot Ollama health probe.

    `ok=True` means the daemon is reachable and the configured model is
    available locally. Otherwise `reason` carries a human-readable string
    and `hint` carries a one-line shell-command remediation."""

    ok: bool
    reason: str = ""
    hint: str = ""


def health_check(model: str = MODEL) -> HealthReport:
    """Probe the local Ollama daemon and the configured model.

    Returns a HealthReport instead of raising — the caller decides how
    to surface failures (chat message, log line, exit code). Each failure
    mode comes with a one-line shell hint that fixes it."""
    try:
        listing = ollama.list()
    except Exception as e:
        return HealthReport(
            ok=False,
            reason=f"Ollama daemon unreachable ({type(e).__name__}: {e}).",
            hint="ollama serve &",
        )

    # ollama.list() returns {'models': [{'model': 'gemma4:e4b', ...}, ...]}
    # The field can be 'model' or 'name' depending on client version.
    names = {
        m.get("model") or m.get("name") or ""
        for m in (listing.get("models") or [])
    }
    if model not in names:
        return HealthReport(
            ok=False,
            reason=f"Model `{model}` not pulled (have: {sorted(names) or 'nothing'}).",
            hint=f"ollama pull {model}",
        )

    return HealthReport(ok=True)


def _encode_media(src: str | bytes | Path) -> str:
    """Normalize any media input to a base64 string."""
    if isinstance(src, (bytes, bytearray)):
        return base64.b64encode(bytes(src)).decode()
    if isinstance(src, Path) or (isinstance(src, str) and Path(src).exists()):
        return base64.b64encode(Path(src).read_bytes()).decode()
    return str(src)


def stream_chat(
    question: str,
    images: Sequence[str | bytes | Path] = (),
    *,
    history: list[dict] | None = None,
    audio: bytes | None = None,
    num_ctx: int = 24_000,
    model: str = MODEL,
) -> Iterator[str]:
    """Stream a Gemma 4 response chunk-by-chunk.

    `images` covers PNG/JPEG bytes, file paths, or base64 strings.
    `audio` is raw WAV bytes; it's appended to the same `images` list
    because Ollama uses that field for all binary media.

    `num_ctx` defaults to 24K — a deliberate compromise that fits
    several whole research-paper pages on an 8GB GPU without VRAM
    spilling to RAM. Bump it explicitly when you need more.
    """
    payload: list[str] = [_encode_media(i) for i in images]
    if audio is not None:
        payload.append(_encode_media(audio))

    user_msg: dict = {"role": "user", "content": question}
    if payload:
        user_msg["images"] = payload

    messages = list(history or []) + [user_msg]
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"num_ctx": num_ctx},
    )
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta
