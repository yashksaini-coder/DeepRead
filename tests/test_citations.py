from __future__ import annotations

from deepread.citations import citation_prompt, extract_citations
from deepread.ingest import Shard


def _shard(cid: str) -> Shard:
    return Shard(cite_id=cid, source="x", page=1, png_bytes=b"", extracted_text="")


def test_empty_catalog_returns_a_helpful_prompt():
    prompt = citation_prompt([])
    assert "ask the user to add" in prompt.lower()


def test_catalog_lists_every_cite_id_exactly_once():
    shards = [_shard("a.pdf#p1"), _shard("a.pdf#p2"), _shard("chart.png")]
    prompt = citation_prompt(shards)
    for s in shards:
        assert f"- {s.cite_id}" in prompt
    # The instruction text appears verbatim
    assert "[[cite_id]]" in prompt


def test_extract_citations_finds_single_inline_id():
    answer = "The figure on page 8 shows growth [[a.pdf#p8]]."
    assert extract_citations(answer) == ["a.pdf#p8"]


def test_extract_citations_deduplicates_preserving_order():
    answer = (
        "Both [[a.pdf#p3]] and [[b.pdf#p1]] support this. "
        "See also [[a.pdf#p3]]."
    )
    assert extract_citations(answer) == ["a.pdf#p3", "b.pdf#p1"]


def test_extract_citations_handles_zero_citations():
    assert extract_citations("Just a plain answer with no marks.") == []


def test_extract_citations_ignores_newlines_inside_brackets():
    """Real-world model output sometimes wraps awkwardly — the regex
    intentionally rejects citations spanning newlines so we don't
    glue unrelated content together."""
    answer = "broken [[a.pdf\n#p1]] not a citation."
    assert extract_citations(answer) == []


def test_single_brackets_accepted_when_id_is_known():
    """Models sometimes emit [id] instead of [[id]]. We accept the
    single-bracket form ONLY when the id matches a known shard, so
    we don't accidentally treat plain markdown like [1] as citations."""
    answer = "This is supported [a.pdf#p3] but [1] is a list marker."
    known = {"a.pdf#p3"}
    assert extract_citations(answer, known=known) == ["a.pdf#p3"]


def test_single_brackets_rejected_when_id_is_unknown():
    answer = "Bad guess [missing.pdf#p9] should not count."
    known = {"a.pdf#p1"}
    assert extract_citations(answer, known=known) == []


def test_double_brackets_always_accepted_even_without_known_set():
    answer = "Solid citation [[a.pdf#p3]]."
    assert extract_citations(answer) == ["a.pdf#p3"]
