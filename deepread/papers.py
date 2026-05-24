"""Curated example research papers — now bundled locally.

The five papers ship in the repo's `papers/` directory (~3.3 MB total).
On click, they load instantly from disk — no network call, no
toast-narrated download. The "0 bytes sent to cloud" claim becomes
literally true at runtime.

`scripts/refresh_papers.py` re-downloads them from canonical URLs for
maintenance; the user-facing code never touches the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# REPO_PAPERS_DIR is the bundled location. CACHE_DIR is the legacy
# download fallback — only used when refresh_papers.py runs.
REPO_PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers_pdf_download"
CACHE_DIR = Path(
    os.environ.get("DEEPREAD_CACHE", Path.home() / ".cache" / "deepread")
) / "papers"


@dataclass(slots=True, frozen=True)
class ExamplePaper:
    """One curated paper that ships with the repo."""

    slug: str           # cache filename basename, [a-z0-9-]+
    title: str          # full title — used in toasts + accessibility
    short: str          # ≤14-char shorthand for the card UI
    authors: str        # display attribution
    year: int
    url: str            # canonical URL — for refresh_papers.py only
    pages: int          # exact, matches the bundled PDF
    blurb: str          # one-line "why this paper matters" tooltip

    @property
    def cache_path(self) -> Path:
        """Path to the cached/bundled PDF. Prefers the in-repo `papers/`
        copy; falls back to ~/.cache/deepread/papers/ for legacy/dev."""
        bundled = REPO_PAPERS_DIR / f"{self.slug}.pdf"
        if bundled.exists() and bundled.stat().st_size > 0:
            return bundled
        return CACHE_DIR / f"{self.slug}.pdf"

    @property
    def cached(self) -> bool:
        p = self.cache_path
        return p.exists() and p.stat().st_size > 0


# Order matters — this is the visual order in the sidebar.
EXAMPLE_PAPERS: list[ExamplePaper] = [
    ExamplePaper(
        slug="attention",
        title="Attention Is All You Need",
        short="Attention",
        authors="Vaswani et al.",
        year=2017,
        url="https://arxiv.org/pdf/1706.03762",
        pages=15,
        blurb="The Transformer architecture paper — foundation of GPT, Gemma, and most modern LLMs.",
    ),
    ExamplePaper(
        slug="gfs",
        title="The Google File System",
        short="GFS",
        authors="Ghemawat et al.",
        year=2003,
        url="https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf",
        pages=15,
        blurb="The distributed-systems classic that inspired HDFS and a generation of storage infrastructure.",
    ),
    ExamplePaper(
        slug="mapreduce",
        title="MapReduce: Simplified Data Processing on Large Clusters",
        short="MapReduce",
        authors="Dean & Ghemawat",
        year=2004,
        url="https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf",
        pages=13,
        blurb="The programming model that turned commodity clusters into approachable infrastructure.",
    ),
    ExamplePaper(
        slug="raft",
        title="In Search of an Understandable Consensus Algorithm (Raft)",
        short="Raft",
        authors="Ongaro & Ousterhout",
        year=2014,
        url="https://raft.github.io/raft.pdf",
        pages=18,
        blurb="Distributed consensus, deliberately designed to be readable.",
    ),
    ExamplePaper(
        slug="bitcoin",
        title="Bitcoin: A Peer-to-Peer Electronic Cash System",
        short="Bitcoin",
        authors="Nakamoto",
        year=2008,
        url="https://bitcoin.org/bitcoin.pdf",
        pages=9,
        blurb="Nine pages that started a new field. Heavy on diagrams — good vision test.",
    ),
]

PAPERS_BY_SLUG: dict[str, ExamplePaper] = {p.slug: p for p in EXAMPLE_PAPERS}


def get(slug: str) -> ExamplePaper:
    if slug not in PAPERS_BY_SLUG:
        raise KeyError(f"unknown example paper: {slug!r}")
    return PAPERS_BY_SLUG[slug]


def download(
    paper: ExamplePaper,
    on_progress: Callable[[int, int | None], None] | None = None,
    *,
    timeout: float = 30.0,
) -> Path:
    """Return the bundled PDF path if present; otherwise download it.

    Hot-path callers should never trigger the network branch — the
    `papers/` directory ships pre-populated. The download fallback is
    only reached if someone deletes the bundle, in which case we treat
    it as a recovery step rather than a normal interaction.
    """
    if paper.cached:
        if on_progress is not None:
            size = paper.cache_path.stat().st_size
            on_progress(size, size)
        return paper.cache_path

    # Recovery fallback — only triggered when the bundle is missing.
    import httpx

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = (CACHE_DIR / f"{paper.slug}.pdf").with_suffix(".pdf.partial")
    with httpx.stream(
        "GET", paper.url, follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "DeepRead/0.1"},
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0) or None
        downloaded = 0
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=32_768):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)
    tmp.replace(CACHE_DIR / f"{paper.slug}.pdf")
    return CACHE_DIR / f"{paper.slug}.pdf"
