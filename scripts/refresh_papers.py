"""Re-download the 5 bundled example papers from canonical URLs.

Run this once if `papers/` is missing or stale. Normal user flow never
needs to invoke this — the demo ships with the PDFs pre-bundled so
nothing on the hot path touches the network.

    uv run python scripts/refresh_papers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx

from deepread.papers import EXAMPLE_PAPERS, REPO_PAPERS_DIR


def main() -> int:
    REPO_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for p in EXAMPLE_PAPERS:
        target = REPO_PAPERS_DIR / f"{p.slug}.pdf"
        print(f"  {p.slug:12s} → {p.url}")
        with httpx.stream(
            "GET", p.url, follow_redirects=True, timeout=60,
            headers={"User-Agent": "DeepRead-refresh/0.1"},
        ) as resp:
            resp.raise_for_status()
            tmp = target.with_suffix(".pdf.partial")
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=64_000):
                    f.write(chunk)
            tmp.replace(target)
        size_kb = target.stat().st_size // 1024
        print(f"  {' ' * 12}   {size_kb} KB")
    print(f"\n{len(EXAMPLE_PAPERS)} papers refreshed in {REPO_PAPERS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
