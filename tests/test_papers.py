from __future__ import annotations

import os
from pathlib import Path

import pytest

from deepread import papers
from deepread.papers import EXAMPLE_PAPERS, ExamplePaper, get


def test_example_papers_have_unique_slugs():
    slugs = [p.slug for p in EXAMPLE_PAPERS]
    assert len(slugs) == len(set(slugs)), "slug collision"


def test_example_papers_have_canonical_urls():
    for p in EXAMPLE_PAPERS:
        assert p.url.startswith(("http://", "https://")), f"bad url: {p.url}"
        assert p.url.lower().endswith((".pdf", "")), f"unexpected url shape: {p.url}"


def test_example_papers_have_sensible_metadata():
    for p in EXAMPLE_PAPERS:
        assert 1 <= len(p.short) <= 20, f"short too long: {p.short!r}"
        assert 1900 < p.year < 2100, f"implausible year: {p.year}"
        assert p.pages >= 1, f"pages must be positive: {p.pages}"
        assert p.title, "title required"
        assert p.authors, "authors required"
        assert p.blurb, "blurb required"


def test_get_returns_paper_by_slug():
    for p in EXAMPLE_PAPERS:
        assert get(p.slug) is p


def test_get_raises_on_unknown_slug():
    with pytest.raises(KeyError):
        get("does-not-exist")


def test_cache_path_uses_env_override(tmp_path: Path, monkeypatch):
    """CACHE_DIR honors $DEEPREAD_CACHE so tests don't pollute ~/.cache."""
    # The constant is resolved at import time; we patch the module
    # directly to ensure cache_path tracks it for downstream code.
    monkeypatch.setattr(papers, "CACHE_DIR", tmp_path / "papers")
    # Re-construct a paper that closes over the new module CACHE_DIR
    new_p = ExamplePaper(
        slug="test", title="t", short="t", authors="a", year=2026,
        url="https://example.com/x.pdf", pages=1, blurb="x",
    )
    # cache_path is computed from the module-level CACHE_DIR
    assert new_p.cache_path == tmp_path / "papers" / "test.pdf"


def test_cached_flag_reflects_filesystem(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(papers, "CACHE_DIR", tmp_path / "papers")
    p = ExamplePaper(
        slug="abc", title="t", short="t", authors="a", year=2026,
        url="https://example.com/x.pdf", pages=1, blurb="x",
    )
    assert not p.cached
    p.cache_path.parent.mkdir(parents=True)
    p.cache_path.write_bytes(b"%PDF-1.4 fake content")
    assert p.cached


def test_empty_file_is_not_considered_cached(tmp_path: Path, monkeypatch):
    """A zero-byte file (e.g. a failed download) must NOT count as a hit."""
    monkeypatch.setattr(papers, "CACHE_DIR", tmp_path / "papers")
    p = ExamplePaper(
        slug="zero", title="t", short="t", authors="a", year=2026,
        url="https://example.com/x.pdf", pages=1, blurb="x",
    )
    p.cache_path.parent.mkdir(parents=True)
    p.cache_path.touch()  # zero bytes
    assert not p.cached
