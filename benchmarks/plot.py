"""Render benchmark/results.json into a chart for Article 1.

Outputs `benchmarks/plot.png` showing pass-rate, tokens/sec, and
time-to-first-token across the swept context lengths.

Run: `uv run python benchmarks/plot.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results.json"
OUT = Path(__file__).parent / "plot.png"


def load() -> list[dict]:
    """Filter to clean data points: 5 needles AND ≥10K context.

    The 5K-context-with-5-needles row is excluded — its 60% pass rate
    is a calibration artifact of asking 5 needles to spread across a
    context too small to hold them with breathing room. See the
    `--ctx 5000 --needles 3` row in results.json for a clean 5K data
    point.
    """
    rows = []
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" in r:
                continue
            if r.get("needles", 0) >= 5 and r.get("target_ctx", 0) >= 10_000:
                rows.append(r)
    rows.sort(key=lambda r: r["target_ctx"])
    return rows


def main() -> None:
    rows = load()
    if not rows:
        print("No benchmark rows to plot.")
        return

    ctx = [r["target_ctx"] / 1000 for r in rows]
    pass_rate = [r["pass_rate"] * 100 for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]
    ttft = [r["first_token_seconds"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ctx, pass_rate, "-o", color="#16a34a", linewidth=2)
    axes[0].set_title("Needle-in-Haystack Pass Rate")
    axes[0].set_xlabel("Context length (K tokens)")
    axes[0].set_ylabel("Pass rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ctx, tps, "-o", color="#2563eb", linewidth=2)
    axes[1].set_title("Generation Throughput")
    axes[1].set_xlabel("Context length (K tokens)")
    axes[1].set_ylabel("Tokens / sec")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ctx, ttft, "-o", color="#dc2626", linewidth=2)
    axes[2].set_title("Time to First Token")
    axes[2].set_xlabel("Context length (K tokens)")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Gemma 4 E4B on RTX 5050 (8 GB) · Ollama 0.24.0", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
