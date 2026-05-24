"""Context-length sweep for Gemma 4 — feeds Article 1.

For each (model, target context length) pair we measure:

  - **tokens/sec** during generation (proxy for user-facing latency)
  - **wall time** to first token and to completion
  - **needle-in-haystack pass-rate**: K specific facts are seeded at
    different positions in a long synthetic document; we ask each one
    back and grade exact-match recovery.

The script writes a JSON line per (model, ctx) cell to
`benchmarks/results.json`. Resume is trivial — re-running the script
appends new rows; downstream charts dedupe on (model, ctx).

Runtime budget: full sweep on this laptop is ~25 min per model. Run
overnight with `uv run python benchmarks/run_context_sweep.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import ollama

RESULTS = Path(__file__).parent / "results.json"

# Seed the haystack with English-readable filler so the model can
# operate on it as ordinary text. Each filler "passage" is ~80 tokens.
FILLER = (
    "The river is wide and brown, and the bridge above it carries trains in "
    "both directions throughout the day. Pedestrians use the lower walkway, "
    "which has benches every twenty metres. Vendors sell roasted peanuts and "
    "tea at the south end, near the fountain. The fountain was added in the "
    "1970s and is illuminated at night."
)


@dataclass(slots=True)
class Needle:
    code: str   # e.g. "K7XQ"
    fact: str   # e.g. "The secret pass code K7XQ unlocks vault 4."
    position_pct: float  # 0..1 — where in the context this needle was placed
    question: str


def make_needles(k: int = 5) -> list[Needle]:
    rng = random.Random(20260521)
    needles = []
    for i, pos in enumerate([0.05, 0.25, 0.50, 0.75, 0.95][:k]):
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
        needles.append(
            Needle(
                code=code,
                fact=f"REMEMBER: The unique pass code for box {i + 1} is {code}.",
                position_pct=pos,
                question=f"What is the unique pass code for box {i + 1}?",
            )
        )
    return needles


def build_haystack(target_tokens: int, needles: list[Needle]) -> str:
    """Build a roughly `target_tokens`-tokens haystack with needles
    interleaved at their target positions."""
    chunks_needed = max(target_tokens * 4 // len(FILLER), 100)
    chunks = [FILLER] * chunks_needed
    for n in needles:
        idx = int(len(chunks) * n.position_pct)
        chunks[idx] = chunks[idx] + " " + n.fact
    return " ".join(chunks)


def measure(model: str, ctx: int, needle_count: int = 5) -> dict:
    needles = make_needles(needle_count)
    haystack = build_haystack(ctx, needles)
    # Sanity: don't exceed model context budget for the prompt itself.
    haystack_chars = min(len(haystack), ctx * 4)
    haystack = haystack[:haystack_chars]

    passed = 0
    per_needle: list[dict] = []
    total_gen_tokens = 0
    total_gen_seconds = 0.0
    first_token_seconds: float | None = None

    for n in needles:
        prompt = (
            f"You will be given a long document. Several special facts are "
            f"hidden inside it, each beginning with the word REMEMBER. "
            f"Then a question follows. Answer it from the document only.\n\n"
            f"<document>\n{haystack}\n</document>\n\n"
            f"Question: {n.question}\nAnswer with just the code."
        )

        t0 = time.perf_counter()
        first_t = None
        out = ""
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_ctx": ctx + 2000, "temperature": 0.0},
        ):
            delta = chunk.get("message", {}).get("content", "")
            if delta and first_t is None:
                first_t = time.perf_counter()
            out += delta
        t1 = time.perf_counter()
        gen_tokens = max(1, len(out) // 4)  # crude tokens estimate
        gen_seconds = (t1 - (first_t or t0))
        total_gen_tokens += gen_tokens
        total_gen_seconds += gen_seconds
        if first_token_seconds is None and first_t is not None:
            first_token_seconds = first_t - t0

        ok = n.code in out.upper()
        passed += int(ok)
        per_needle.append(
            {"code": n.code, "pos": n.position_pct, "ok": ok, "got": out.strip()[:80]}
        )

    tps = (total_gen_tokens / total_gen_seconds) if total_gen_seconds > 0 else 0
    return {
        "model": model,
        "target_ctx": ctx,
        "haystack_chars": haystack_chars,
        "needles": needle_count,
        "passed": passed,
        "pass_rate": round(passed / needle_count, 2),
        "tokens_per_sec": round(tps, 2),
        "first_token_seconds": round(first_token_seconds or 0, 2),
        "per_needle": per_needle,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma4:e4b"],
        help="Ollama model tags to sweep",
    )
    parser.add_argument(
        "--ctx",
        nargs="+",
        type=int,
        default=[5_000, 20_000, 60_000, 100_000],
        help="Target context lengths in tokens (E2B/E4B max 128K)",
    )
    parser.add_argument("--needles", type=int, default=5)
    args = parser.parse_args()

    out = RESULTS.open("a")
    for model in args.models:
        for ctx in args.ctx:
            print(f">>> {model} @ {ctx:,} tokens — measuring…")
            try:
                row = measure(model, ctx, args.needles)
            except Exception as e:
                row = {
                    "model": model,
                    "target_ctx": ctx,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            print(json.dumps(row, indent=2))
            out.write(json.dumps(row) + "\n")
            out.flush()
    out.close()


if __name__ == "__main__":
    main()
