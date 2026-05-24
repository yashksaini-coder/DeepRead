"""End-to-end smoke test: text-only chat against gemma4:e4b.

Run: `uv run python scripts/smoke_text.py`
"""

from __future__ import annotations

import sys
import time

from deepread.llm import stream_chat


def main() -> int:
    print(">>> gemma4:e4b text smoke test")
    t0 = time.time()
    out: list[str] = []
    for delta in stream_chat(
        "In one sentence, what is the capital of France?",
        num_ctx=4096,
    ):
        sys.stdout.write(delta)
        sys.stdout.flush()
        out.append(delta)
    elapsed = time.time() - t0
    text = "".join(out)
    print(f"\n--- {elapsed:.1f}s, {len(text)} chars ---")
    return 0 if "Paris" in text else 1


if __name__ == "__main__":
    sys.exit(main())
