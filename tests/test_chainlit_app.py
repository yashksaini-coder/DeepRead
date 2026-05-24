"""Tests for the Chainlit app's pure-Python helpers.

Chainlit handlers run inside a WebSocket session and are awkward to
unit-test in isolation — they depend on a live Chainlit context to call
`cl.user_session.get/set` and `cl.Message().send()`. So this test file
focuses on the PURE helpers we extracted (no Chainlit-context required):
budget summary formatting, citation formatting, working-set logic.

The actual end-to-end handler behavior is verified via the live
browser smoke test in `scripts/chainlit_smoke.py` (manual) and via
the existing 27 deepread/* unit tests (which underlie everything).
"""

from __future__ import annotations

from deepread.ingest import Shard


def _shard(source: str, page: int = 1) -> Shard:
    """Cheap synthetic Shard for tests — no PDF needed."""
    return Shard(
        cite_id=f"{source}#p{page}",
        source=source,
        page=page,
        png_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,  # minimal PNG header
        extracted_text=f"page {page} of {source}",
    )


# ── working_set logic ─────────────────────────────────────────────────


def test_working_set_excludes_listed_sources():
    from app import _working_set

    shards = [_shard("a.pdf", 1), _shard("a.pdf", 2), _shard("b.pdf", 1)]
    excluded = {"a.pdf"}
    ws = _working_set(shards, excluded)
    assert len(ws) == 1
    assert ws[0].source == "b.pdf"


def test_working_set_includes_all_when_excluded_is_empty():
    from app import _working_set

    shards = [_shard("a.pdf", 1), _shard("b.pdf", 1)]
    assert len(_working_set(shards, set())) == 2


# ── budget summary ────────────────────────────────────────────────────


def test_budget_summary_empty_library():
    from app import _budget_summary

    out = _budget_summary([], set())
    assert "0.0%" in out
    assert "128,000" in out
    assert "interactive" in out


def test_budget_summary_reports_pct_and_bar():
    from app import _budget_summary

    # 10 image shards × 900 tok/img ≈ 9,000 tokens → ~7% of 128k
    shards = [_shard(f"a.pdf", i) for i in range(1, 11)]
    out = _budget_summary(shards, set())
    assert "7" in out  # ~7% somewhere in the readout
    assert "9,000" in out


def test_budget_summary_marks_over_limit_visually():
    from app import _budget_summary
    from deepread.budget import CONTEXT_LIMIT, TOKENS_PER_IMAGE

    # Enough image shards to blow past the 128K ceiling
    n = (CONTEXT_LIMIT // TOKENS_PER_IMAGE) + 5
    shards = [_shard("big.pdf", i) for i in range(1, n + 1)]
    out = _budget_summary(shards, set())
    assert "over limit" in out


# ── citation formatting ───────────────────────────────────────────────


def test_format_answer_extracts_double_bracket_citations_in_order():
    from app import _format_answer

    known = {"paper.pdf#p1", "paper.pdf#p3"}
    body, cites = _format_answer(
        "First claim [[paper.pdf#p1]]. Second claim [[paper.pdf#p3]].",
        known,
    )
    assert cites == ["paper.pdf#p1", "paper.pdf#p3"]
    assert "[^1]" in body and "[^2]" in body


def test_format_answer_deduplicates_repeated_citations():
    from app import _format_answer

    known = {"paper.pdf#p1"}
    body, cites = _format_answer(
        "First [[paper.pdf#p1]] and again [[paper.pdf#p1]].",
        known,
    )
    assert cites == ["paper.pdf#p1"]
    # Both occurrences should be footnote [^1]
    assert body.count("[^1]") == 2


def test_format_answer_accepts_single_bracket_when_known():
    from app import _format_answer

    known = {"paper.pdf#p3"}
    body, cites = _format_answer(
        "Claim [paper.pdf#p3] backed by single brackets.",
        known,
    )
    assert cites == ["paper.pdf#p3"]


def test_format_answer_rejects_single_bracket_when_not_known():
    from app import _format_answer

    known = {"a.pdf#p1"}
    body, cites = _format_answer(
        "Claim [unknown.pdf#p9] not in the library.",
        known,
    )
    assert cites == []


def test_citation_appendix_renders_footnote_definitions():
    from app import _citation_appendix

    out = _citation_appendix(["a.pdf#p3", "b.pdf#p5"])
    assert "[^1]: **a.pdf** · page 3" in out
    assert "[^2]: **b.pdf** · page 5" in out


def test_citation_appendix_empty_returns_empty():
    from app import _citation_appendix

    assert _citation_appendix([]) == ""


# ── PaperPicker custom element props ──────────────────────────────────


def test_picker_props_lists_every_bundled_paper():
    from app import _picker_props
    from deepread.papers import EXAMPLE_PAPERS

    props = _picker_props([])
    assert {p["slug"] for p in props["papers"]} == {p.slug for p in EXAMPLE_PAPERS}
    # Each entry carries the fields the React component reads
    for p in props["papers"]:
        assert {"slug", "short", "year", "pages", "title", "authors", "cache_name"} <= p.keys()
    # No papers loaded yet → empty loaded list
    assert props["loaded"] == []


def test_picker_props_marks_loaded_shards():
    from app import _picker_props

    shards = [_shard("attention.pdf", 1), _shard("bitcoin.pdf", 1)]
    props = _picker_props(shards)
    assert set(props["loaded"]) == {"attention.pdf", "bitcoin.pdf"}


# ── Plotly chart helpers ──────────────────────────────────────────────


def test_budget_chart_returns_plotly_figure_with_correct_xaxis_range():
    from app import _budget_chart
    import plotly.graph_objects as go

    fig = _budget_chart([], set())
    assert isinstance(fig, go.Figure)
    # X-axis must span 0 → CONTEXT_LIMIT (128K)
    xrange = fig.layout.xaxis.range
    assert xrange[0] == 0
    assert xrange[1] == 128_000


def test_budget_chart_color_tracks_tier():
    from app import _budget_chart, _tier_color
    from deepread.budget import CONTEXT_LIMIT, TOKENS_PER_IMAGE

    # Empty library → green (interactive tier)
    fig = _budget_chart([], set())
    assert fig.data[0].marker.color == _tier_color(0)  # green
    # Cross the safe limit → red
    n_red = (CONTEXT_LIMIT // TOKENS_PER_IMAGE) + 1
    shards = [_shard(f"x.pdf", i) for i in range(1, n_red + 1)]
    fig = _budget_chart(shards, set())
    assert fig.data[0].marker.color == _tier_color(n_red * TOKENS_PER_IMAGE)


def test_benchmark_chart_handles_empty_rows():
    from app import _benchmark_chart
    import plotly.graph_objects as go

    fig = _benchmark_chart([])
    assert isinstance(fig, go.Figure)
    # Empty state shows an annotation, not data traces
    assert len(fig.data) == 0
    assert any("No benchmark results yet" in (a.text or "") for a in fig.layout.annotations)


def test_benchmark_chart_renders_three_panel_layout_from_rows():
    from app import _benchmark_chart

    rows = [
        {"target_ctx": 20_000, "pass_rate": 1.0, "tokens_per_sec": 8.6, "first_token_seconds": 15.0, "needles": 5},
        {"target_ctx": 60_000, "pass_rate": 1.0, "tokens_per_sec": 7.6, "first_token_seconds": 38.3, "needles": 5},
        {"target_ctx": 100_000, "pass_rate": 1.0, "tokens_per_sec": 6.8, "first_token_seconds": 72.3, "needles": 5},
    ]
    fig = _benchmark_chart(rows)
    # 3 traces (pass rate, tps, ttft)
    assert len(fig.data) == 3
    # X-axis values should be in K-tokens (20, 60, 100)
    assert list(fig.data[0].x) == [20.0, 60.0, 100.0]


def test_load_benchmark_rows_returns_empty_when_no_file(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "BENCHMARK_RESULTS", tmp_path / "missing.json")
    assert app._load_benchmark_rows() == []


def test_load_benchmark_rows_filters_invalid_rows(tmp_path, monkeypatch):
    import app

    results = tmp_path / "r.json"
    results.write_text(
        '{"target_ctx": 20000, "needles": 5, "pass_rate": 1.0, "tokens_per_sec": 8, "first_token_seconds": 15}\n'
        '{"target_ctx": 5000, "needles": 3, "pass_rate": 1.0, "tokens_per_sec": 8, "first_token_seconds": 12}\n'
        '{"error": "bad row"}\n'
        'not valid json\n'
    )
    monkeypatch.setattr(app, "BENCHMARK_RESULTS", results)
    rows = app._load_benchmark_rows()
    # Only the 20K row passes (needles>=5 AND ctx>=10000)
    assert len(rows) == 1
    assert rows[0]["target_ctx"] == 20_000


# ── Ollama health probe ───────────────────────────────────────────────


def test_health_check_reports_daemon_unreachable(monkeypatch):
    from deepread import llm

    def boom():
        raise ConnectionRefusedError("dial tcp 127.0.0.1:11434: connection refused")
    monkeypatch.setattr(llm.ollama, "list", boom)
    r = llm.health_check("gemma4:e4b")
    assert not r.ok
    assert "unreachable" in r.reason.lower()
    assert r.hint.startswith("ollama serve")


def test_health_check_reports_missing_model(monkeypatch):
    from deepread import llm

    monkeypatch.setattr(
        llm.ollama, "list",
        lambda: {"models": [{"model": "llama3:8b"}, {"model": "phi3:mini"}]},
    )
    r = llm.health_check("gemma4:e4b")
    assert not r.ok
    assert "not pulled" in r.reason
    assert r.hint == "ollama pull gemma4:e4b"


def test_health_check_succeeds_when_model_present(monkeypatch):
    from deepread import llm

    monkeypatch.setattr(
        llm.ollama, "list",
        lambda: {"models": [{"model": "gemma4:e4b"}, {"model": "llama3:8b"}]},
    )
    r = llm.health_check("gemma4:e4b")
    assert r.ok
    assert r.reason == ""


# ── sweep command parser ──────────────────────────────────────────────


def test_parse_sweep_command_returns_none_for_non_bench_text():
    from app import _parse_sweep_command
    assert _parse_sweep_command("hello there") is None
    assert _parse_sweep_command("/bench show") is None  # /bench show isn't a sweep
    assert _parse_sweep_command("run sweep --ctx 5000") is None  # legacy prefix gone


def test_parse_sweep_command_handles_full_flag_set():
    from app import _parse_sweep_command
    args = _parse_sweep_command(
        "/bench run --ctx 5000 20000 60000 --needles 5 --models gemma4:e4b"
    )
    assert args is not None
    assert args["ctx"] == [5_000, 20_000, 60_000]
    assert args["needles"] == 5
    assert args["models"] == ["gemma4:e4b"]


def test_parse_sweep_command_uses_defaults_when_no_flags():
    from app import _parse_sweep_command
    args = _parse_sweep_command("/bench run")
    assert args is not None
    assert args["ctx"] == [5_000, 20_000]    # default zones
    assert args["needles"] == 3              # default needle count


def test_is_bench_command_recognizes_both_prefixes_and_rejects_chat():
    from app import _is_bench_command
    assert _is_bench_command("/bench show")
    assert _is_bench_command("/benchmark run --ctx 5000")
    assert _is_bench_command("/BENCH")               # case-insensitive
    assert _is_bench_command("  /bench  ")           # tolerates whitespace
    # Anything else routes to research-flow document Q&A
    assert not _is_bench_command("What does the paper say?")
    assert not _is_bench_command("show results")     # legacy form, no longer a command
    assert not _is_bench_command("")
