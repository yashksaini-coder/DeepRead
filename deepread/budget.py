"""Token-budget accounting against Gemma 4 E4B's 128K context window.

The budget is the *headline UI feature* — it makes the context window
visible rather than implicit. Constants below are empirical estimates
that get calibrated against real measurements in
`benchmarks/run_context_sweep.py`.
"""

from __future__ import annotations

from .ingest import Shard

TOKENS_PER_IMAGE = 900
TOKENS_PER_TEXT_CHAR = 0.30
CONTEXT_LIMIT = 128_000
SAFE_LIMIT = 110_000


def estimate(shard: Shard) -> int:
    """Estimate the token cost of including this shard in a prompt.

    An image shard's PNG bytes dominate the cost — text extracted from
    the same PDF page is only used for the text-only ingest path.
    """
    if shard.png_bytes:
        return TOKENS_PER_IMAGE
    return int(len(shard.extracted_text) * TOKENS_PER_TEXT_CHAR)


def budget(shards: list[Shard]) -> dict:
    used = sum(estimate(s) for s in shards)
    pct = min(100.0, round(100 * used / CONTEXT_LIMIT, 1))
    return {
        "used": used,
        "limit": CONTEXT_LIMIT,
        "safe": SAFE_LIMIT,
        "pct": pct,
        "over_safe": used > SAFE_LIMIT,
        "over_limit": used > CONTEXT_LIMIT,
    }
