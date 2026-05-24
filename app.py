"""DeepRead — Chainlit frontend.

Local document Q&A powered by Gemma 4 E4B via Ollama. Truly local —
no cloud, no telemetry, no per-query cost. Drop in a PDF (or click a
bundled classic paper), ask anything, get an answer with page-anchored
citations.

The `deepread/` package is UI-framework-independent — it handles PDF
ingestion, token budgeting, citation parsing, and Ollama streaming.
This file is just the Chainlit handlers that wire it to a chat surface.

Run with:
    uv run chainlit run app.py -h --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import chainlit as cl
import plotly.graph_objects as go

logger = logging.getLogger("deepread.app")
logging.basicConfig(
    level=os.environ.get("DEEPREAD_LOG", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from deepread.budget import CONTEXT_LIMIT, SAFE_LIMIT, budget
from deepread.citations import CITATION_RE, citation_prompt
from deepread.ingest import Shard, ingest_image, ingest_pdf
from deepread.llm import MODEL, health_check, stream_chat
from deepread import papers as _papers

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_RESULTS = REPO_ROOT / "benchmarks" / "results.json"

# ─── per-session state keys ──────────────────────────────────────────
# Chainlit's cl.user_session is per-WebSocket-connection. Each browser
# tab has its own shards/excluded. This is the right scope for a
# single-user local tool — different tabs = different libraries.
SHARDS_KEY = "shards"
EXCLUDED_KEY = "excluded"


# ─── helpers ─────────────────────────────────────────────────────────


def _working_set(shards: list[Shard], excluded: set[str]) -> list[Shard]:
    return [s for s in shards if s.source not in excluded]


def _budget_summary(shards: list[Shard], excluded: set[str]) -> str:
    """One-line markdown summary of the current context budget.

    Rendered in the pinned budget message and in toasts after each load.
    Format: `**Context** · 6.3% (8,100 / 128,000 tokens) · *interactive*`.
    """
    ws = _working_set(shards, excluded)
    b = budget(ws)
    pct = b["pct"]
    used = b["used"]

    if b["over_limit"]:
        tier = "over limit ⚠️"
    elif b["over_safe"]:
        tier = "tight"
    elif used < 20_000:
        tier = "interactive"
    elif used < 60_000:
        tier = "research"
    elif used < 100_000:
        tier = "batch"
    else:
        tier = "near ceiling"

    bar_chars = 24
    fill = max(0, min(bar_chars, round(pct / 100 * bar_chars)))
    bar = "█" * fill + "░" * (bar_chars - fill)

    return (
        f"**Context** `{bar}` **{pct:.1f}%**  ·  "
        f"`{used:>7,}` / 128,000 tokens  ·  *{tier}*"
    )


def _format_answer(answer: str, known_cite_ids: set[str]) -> tuple[str, list[str]]:
    """Convert [[cite_id]] markers into superscript-style footnotes,
    return the formatted markdown body + the ordered list of unique cite_ids."""
    seen: dict[str, int] = {}
    citations: list[str] = []

    def repl(m: re.Match[str]) -> str:
        double = m.group("id")
        single = m.group("idSingle")
        cid = (double or single or "").strip()
        if not cid:
            return m.group(0)
        if single is not None and cid not in known_cite_ids:
            return m.group(0)
        if cid not in seen:
            citations.append(cid)
            seen[cid] = len(citations)
        n = seen[cid]
        return f"[^{n}]"

    body = CITATION_RE.sub(repl, answer)
    return body, citations


def _citation_appendix(citations: list[str]) -> str:
    """Markdown footnote definitions appended after the answer."""
    if not citations:
        return ""
    lines = ["\n\n---\n"]
    for n, cid in enumerate(citations, start=1):
        if "#p" in cid:
            src, page = cid.rsplit("#p", 1)
            lines.append(f"[^{n}]: **{src}** · page {page}")
        else:
            lines.append(f"[^{n}]: **{cid}**")
    return "\n".join(lines)


# ─── Plotly chart helpers ────────────────────────────────────────────


# Indigo palette mirrors public/style.css for visual continuity.
_GREEN = "#22C55E"   # ok / interactive / research
_AMBER = "#F59E0B"   # batch
_RED = "#EF4444"     # warn / over-limit
_INDIGO = "#6366F1"  # accent
_INK = "#A1A1A9"
_GRID = "rgba(255,255,255,0.08)"


def _tier_color(used: int) -> str:
    if used >= CONTEXT_LIMIT or used > SAFE_LIMIT:
        return _RED
    if used >= 60_000:
        return _AMBER
    return _GREEN


def _budget_chart(shards: list[Shard], excluded: set[str]) -> go.Figure:
    """A horizontal stacked bar showing context fill against tier zones.

    Sized for the narrow right sidebar (~360px wide) — tier labels are
    suppressed; the 20K/60K/100K tickmarks carry the same information
    without overlapping the bar."""
    ws = _working_set(shards, excluded)
    used = budget(ws)["used"]
    color = _tier_color(used)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[used], y=["context"], orientation="h",
        marker=dict(color=color),
        hovertemplate=f"{used:,} tokens used<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[CONTEXT_LIMIT - used], y=["context"], orientation="h",
        marker=dict(color="rgba(255,255,255,0.05)"),
        hoverinfo="skip", showlegend=False,
    ))
    # Tier boundary gridlines (no labels — they collide in narrow sidebar)
    for boundary in (20_000, 60_000, 100_000):
        fig.add_vline(
            x=boundary, line_width=1, line_dash="dot", line_color=_GRID,
        )
    fig.update_layout(
        barmode="stack",
        height=70, margin=dict(l=4, r=4, t=4, b=20),
        xaxis=dict(
            range=[0, CONTEXT_LIMIT],
            tickvals=[0, 20_000, 60_000, 100_000, 128_000],
            ticktext=["0", "20K", "60K", "100K", "128K"],
            color=_INK, gridcolor=_GRID, tickfont=dict(size=9),
        ),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_INK, size=10),
        showlegend=False,
    )
    return fig


def _load_benchmark_rows() -> list[dict]:
    """Read existing benchmark results from benchmarks/results.json.
    Returns a list of clean (non-error, ≥5 needles, ≥10K context) rows
    suitable for plotting."""
    if not BENCHMARK_RESULTS.exists():
        return []
    rows: list[dict] = []
    for line in BENCHMARK_RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        if r.get("needles", 0) >= 5 and r.get("target_ctx", 0) >= 10_000:
            rows.append(r)
    rows.sort(key=lambda r: r["target_ctx"])
    return rows


def _benchmark_chart(rows: list[dict]) -> go.Figure:
    """Three-panel benchmark figure: pass rate, throughput, time-to-first-token."""
    if not rows:
        # Empty placeholder so the UI doesn't break before any rows exist
        fig = go.Figure()
        fig.add_annotation(
            text="No benchmark results yet. Run a sweep to populate this chart.",
            showarrow=False, font=dict(color=_INK, size=12),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(
            height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    ctx = [r["target_ctx"] / 1000 for r in rows]
    pass_rate = [r["pass_rate"] * 100 for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]
    ttft = [r["first_token_seconds"] for r in rows]

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Needle Pass Rate (%)", "Tokens/sec", "Time to First Token (s)"),
    )
    fig.add_trace(go.Scatter(
        x=ctx, y=pass_rate, mode="lines+markers",
        line=dict(color=_GREEN, width=2), marker=dict(size=8),
        showlegend=False, name="pass",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ctx, y=tps, mode="lines+markers",
        line=dict(color=_INDIGO, width=2), marker=dict(size=8),
        showlegend=False, name="tps",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ctx, y=ttft, mode="lines+markers",
        line=dict(color=_RED, width=2), marker=dict(size=8),
        showlegend=False, name="ttft",
    ), row=1, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(
            title_text="Context (K tokens)", row=1, col=col,
            color=_INK, gridcolor=_GRID,
        )
        fig.update_yaxes(row=1, col=col, color=_INK, gridcolor=_GRID)
    fig.update_yaxes(range=[0, 105], row=1, col=1)

    fig.update_layout(
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_INK, size=11),
    )
    return fig


# ─── Library sidebar (ElementSidebar, native to Chainlit) ────────────


def _picker_props(shards: list[Shard]) -> dict:
    """Props for the PaperPicker custom element. Includes every bundled
    paper + which ones are currently loaded (so the picker can disable
    already-loaded rows). Pure function — no session access."""
    loaded = sorted({s.source for s in shards})
    papers = [
        {
            "slug": p.slug,
            "short": p.short,
            "year": p.year,
            "pages": p.pages,
            "title": p.title,
            "authors": p.authors,
            "cache_name": p.cache_path.name,
        }
        for p in _papers.EXAMPLE_PAPERS
    ]
    return {"papers": papers, "loaded": loaded}


async def _refresh_library_sidebar() -> None:
    """Render budget + paper picker into the right-side ElementSidebar.

    Sidebar layout (top → bottom):
      1. Plotly context-budget bar
      2. One-line budget summary (used / limit · pct · tier)
      3. PaperPicker custom element — clickable rows that fire load_paper
         / upload_pdf actions directly from the sidebar.
    """
    shards: list[Shard] = cl.user_session.get(SHARDS_KEY) or []
    excluded: set[str] = cl.user_session.get(EXCLUDED_KEY) or set()

    ws = _working_set(shards, excluded)
    b = budget(ws)
    used = b["used"]
    if b["over_limit"]:
        tier = "over"
    elif b["over_safe"]:
        tier = "tight"
    elif used < 20_000:
        tier = "interactive"
    elif used < 60_000:
        tier = "research"
    elif used < 100_000:
        tier = "batch"
    else:
        tier = "near ceiling"
    budget_md = f"`{used:>6,}` / 128,000  ·  **{b['pct']:.1f}%**  ·  *{tier}*"

    fig = _budget_chart(shards, excluded)
    await cl.ElementSidebar.set_title("Context")
    await cl.ElementSidebar.set_elements([
        cl.Plotly(name="ctx", figure=fig, display="inline"),
        cl.Text(content=budget_md, name="budget", display="inline"),
        cl.CustomElement(name="PaperPicker", props=_picker_props(shards), display="inline"),
    ])


# ─── lifecycle handlers ──────────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start() -> None:
    """Single session — document Q&A is the primary flow; `/bench` slash
    commands let the same chat run benchmarks without losing history."""
    cl.user_session.set(SHARDS_KEY, [])
    cl.user_session.set(EXCLUDED_KEY, set())

    await cl.Message(
        content=(
            f"**DeepRead** · local doc Q&A · `{MODEL}`\n\n"
            "Pick a paper from the right sidebar (or upload your own). "
            "Then ask anything — I'll cite the page I answered from.\n\n"
            "_Diagnostic: type `/bench show` to view the latest "
            "context-window benchmark, or `/bench run --ctx 5000 20000 "
            "--needles 3` to kick off a fresh sweep._"
        ),
    ).send()
    # Floating "Context" toggle (CSS-positioned top-right). Sending it as a
    # dedicated message with author="sidebar_toggle" lets the stylesheet
    # pin its single action button to the chrome.
    await cl.Message(
        content="",
        actions=[
            cl.Action(
                name="open_sidebar", payload={},
                label="Context", icon="panel-right",
                tooltip="Open the context + papers sidebar",
            ),
        ],
        author="sidebar_toggle",
    ).send()
    await _refresh_library_sidebar()


async def _refresh_budget() -> None:
    """Re-render the right sidebar (budget summary + Plotly + picker).
    The picker auto-updates its disabled-row state from props.loaded."""
    await _refresh_library_sidebar()


# ─── Benchmark mode handlers ─────────────────────────────────────────


async def _send_benchmark_charts(rows: list[dict], *, header: str | None = None) -> None:
    """Send the 3-panel Plotly chart + a compact tabular summary."""
    fig = _benchmark_chart(rows)
    table_lines = ["| Context | Pass rate | Tok/s | TTFT |", "|---:|:---:|---:|---:|"]
    for r in rows:
        ctx = r["target_ctx"]
        table_lines.append(
            f"| {ctx:,} | {r['pass_rate']*100:.0f}% | "
            f"{r['tokens_per_sec']:.1f} | {r['first_token_seconds']:.1f}s |"
        )
    body = (header + "\n\n" if header else "") + "\n".join(table_lines)
    await cl.Message(
        content=body,
        elements=[cl.Plotly(name="benchmark", figure=fig, display="inline", size="large")],
        author="benchmark",
    ).send()


_SWEEP_ARG_RE = re.compile(r"--(\w+)\s+([^-][\w\s,]*?)(?=\s+--|\s*$)")


def _parse_sweep_command(text: str) -> dict | None:
    """Parse `/bench run --ctx 5000 20000 --needles 5` style commands.
    Returns None if the text doesn't start with '/bench run'."""
    text = text.strip().lower()
    if not text.startswith("/bench run"):
        return None
    args = {"ctx": [5_000, 20_000], "needles": 3, "models": ["gemma4:e4b"]}
    for k, v in _SWEEP_ARG_RE.findall(text):
        v = v.strip()
        if k == "ctx":
            args["ctx"] = [int(x.strip()) for x in v.split() if x.strip().isdigit()]
        elif k == "needles":
            try:
                args["needles"] = int(v.split()[0])
            except (ValueError, IndexError):
                pass
        elif k in ("model", "models"):
            args["models"] = [m.strip() for m in v.split() if m.strip()]
    return args


def _is_bench_command(text: str) -> bool:
    """True if the message is a /bench (or /benchmark) slash command."""
    head = text.strip().lower().split(maxsplit=1)
    return bool(head) and head[0] in ("/bench", "/benchmark")


def _run_sweep_blocking(argv: list[str]) -> str:
    """Run the sweep CLI synchronously. Called via asyncio.to_thread so
    the Chainlit event loop stays responsive. argv is a list — no shell.

    Uses sys.executable so the same Python that runs Chainlit also runs
    the sweep — keeps virtualenvs and conda envs working out of the box."""
    import subprocess
    logger.info("sweep starting: %s", " ".join(argv[1:]))
    result = subprocess.run(
        argv, cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
        },
        capture_output=True, text=True, timeout=1800,
    )
    logger.info("sweep finished: rc=%s", result.returncode)
    return (result.stdout + result.stderr)[-3000:]


async def _handle_benchmark_message(message: cl.Message) -> None:
    """Route /bench [show|run ...] subcommands.

    Caller guarantees the text begins with /bench (or /benchmark). We
    normalize to /bench, then dispatch on the next token."""
    text = (message.content or "").strip().lower()
    # Normalize "/benchmark" → "/bench" so both prefixes work.
    if text.startswith("/benchmark"):
        text = "/bench" + text[len("/benchmark"):]
    rest = text[len("/bench"):].strip()  # everything after /bench

    # `/bench`, `/bench show`, `/bench show results` → render existing
    if rest in ("", "show", "show results", "results"):
        rows = _load_benchmark_rows()
        if rows:
            await _send_benchmark_charts(rows, header=f"{len(rows)} runs on file:")
        else:
            await cl.Message(
                content=(
                    "No benchmark results on file yet. Try "
                    "`/bench run --ctx 5000 20000 --needles 3`."
                ),
                author="benchmark",
            ).send()
        return

    # `/bench run ...` → kick off a sweep
    if rest.startswith("run"):
        args = _parse_sweep_command(text)
        if args is None:
            await cl.Message(
                content=(
                    "Couldn't parse sweep flags. Example: "
                    "`/bench run --ctx 5000 20000 --needles 5`"
                ),
                author="benchmark",
            ).send()
            return

        argv = [
            sys.executable, "benchmarks/run_context_sweep.py",
            "--models", *args["models"],
            "--ctx", *[str(c) for c in args["ctx"]],
            "--needles", str(args["needles"]),
        ]
        await cl.Message(
            content=(
                f"Starting sweep: `{' '.join(argv[2:])}`\n\n"
                "This can take several minutes at large context sizes."
            ),
            author="benchmark",
        ).send()

        async with cl.Step(name="run_context_sweep.py", type="tool") as step:
            try:
                output = await asyncio.to_thread(_run_sweep_blocking, argv)
                step.output = output
            except Exception as e:
                step.output = f"Sweep failed: {type(e).__name__}: {e}"
                await cl.Message(content=f"Sweep failed: `{e}`", author="benchmark").send()
                return

        rows = _load_benchmark_rows()
        await _send_benchmark_charts(rows, header=f"Sweep complete. {len(rows)} runs on file:")
        return

    # Unknown subcommand
    await cl.Message(
        content=(
            "Benchmark commands:\n"
            "- `/bench show` — render charts from existing results.json\n"
            "- `/bench run --ctx 5000 20000 60000 --needles 5` — kick off a fresh sweep\n"
            f"\nDefaults: 3 needles, model `{MODEL}`."
        ),
        author="benchmark",
    ).send()


@cl.action_callback("open_sidebar")
async def on_open_sidebar(action: cl.Action) -> None:
    """Re-render the right sidebar — works regardless of whether the user
    previously closed it, because Chainlit's set_elements implicitly opens.
    Sidebar state isn't persisted, so this always succeeds."""
    await _refresh_library_sidebar()


@cl.action_callback("load_paper")
async def on_load_paper(action: cl.Action) -> None:
    slug = action.payload.get("slug", "")
    try:
        paper = _papers.get(slug)
    except KeyError:
        await cl.Message(content=f"Unknown paper: `{slug}`", author="status").send()
        return

    shards: list[Shard] = cl.user_session.get(SHARDS_KEY) or []
    already_loaded = {s.source for s in shards}
    if paper.cache_path.name in already_loaded:
        await cl.Message(
            content=f"**{paper.short}** is already in the library.",
            author="status",
        ).send()
        return

    if not paper.cached:
        await cl.Message(
            content=(
                f"**{paper.short}** isn't bundled here. "
                f"Run `uv run python scripts/refresh_papers.py` to fetch it."
            ),
            author="status",
        ).send()
        return

    # Ingest is sync + can take a moment for large PDFs — show a step
    # so the user knows something's happening.
    async with cl.Step(name=f"Loading {paper.short}", type="tool") as step:
        new_shards = ingest_pdf(paper.cache_path)
        step.output = f"Ingested {len(new_shards)} pages."

    shards.extend(new_shards)
    cl.user_session.set(SHARDS_KEY, shards)

    b = budget(_working_set(shards, cl.user_session.get(EXCLUDED_KEY) or set()))
    await cl.Message(
        content=(
            f"Loaded **{paper.short}** · {len(new_shards)} pages · "
            f"{b['pct']:.0f}% of context."
        ),
        author="status",
    ).send()
    await _refresh_budget()


@cl.action_callback("upload_pdf")
async def on_upload_pdf(action: cl.Action) -> None:
    files = await cl.AskFileMessage(
        content="Upload PDFs or images. Up to 5 files, 20 MB each.",
        accept={
            "application/pdf": [".pdf"],
            "image/png": [".png"],
            "image/jpeg": [".jpg", ".jpeg"],
            "image/webp": [".webp"],
        },
        max_size_mb=20,
        max_files=5,
        timeout=300,
    ).send()
    if not files:
        return

    shards: list[Shard] = cl.user_session.get(SHARDS_KEY) or []
    added_pages = 0
    added_names: list[str] = []

    for f in files:
        path = Path(f.path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                new_shards = ingest_pdf(path)
                shards.extend(new_shards)
                added_pages += len(new_shards)
                added_names.append(f.name)
            elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                # ingest_image expects the file to exist at its path; copy
                # to a stable location since Chainlit's tmp path can vary.
                shards.append(ingest_image(path))
                added_names.append(f.name)
        except Exception as e:  # surfaced to the user — don't swallow
            await cl.Message(
                content=f"Couldn't ingest **{f.name}**: `{e}`",
                author="status",
            ).send()

    cl.user_session.set(SHARDS_KEY, shards)
    if added_names:
        b = budget(_working_set(shards, cl.user_session.get(EXCLUDED_KEY) or set()))
        await cl.Message(
            content=(
                f"Loaded {', '.join(added_names)}"
                + (f" · {added_pages} pages" if added_pages else "")
                + f" · {b['pct']:.0f}% of context."
            ),
            author="status",
        ).send()
    await _refresh_budget()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # `/bench ...` slash commands take precedence over the document Q&A
    # flow — they share the same chat session so history is preserved.
    if _is_bench_command(message.content or ""):
        await _handle_benchmark_message(message)
        return

    shards: list[Shard] = cl.user_session.get(SHARDS_KEY) or []
    excluded: set[str] = cl.user_session.get(EXCLUDED_KEY) or set()
    ws = _working_set(shards, excluded)

    if not ws:
        await cl.Message(
            content=(
                "No documents loaded yet — pick a paper from the **right "
                "sidebar** (or upload your own), then ask again."
            ),
            actions=[
                cl.Action(
                    name="open_sidebar", payload={},
                    label="Open sidebar", icon="panel-right",
                ),
            ],
        ).send()
        return

    sys_msg = {"role": "system", "content": citation_prompt(ws)}
    images = [s.png_bytes for s in ws]
    known_cite_ids = {s.cite_id for s in ws}

    # Streaming answer.
    msg = cl.Message(content="")
    await msg.send()

    t0 = time.time()
    first_token_t: float | None = None
    raw = ""

    # Pre-flight check — a typed HealthReport beats a bare exception when
    # the daemon isn't running or the model isn't pulled. Caching at the
    # session level means we only probe once per browser tab.
    if not cl.user_session.get("ollama_ok"):
        report = await asyncio.to_thread(health_check, MODEL)
        if not report.ok:
            msg.content = (
                f"**Ollama can't serve `{MODEL}` right now.**\n\n"
                f"_{report.reason}_\n\n"
                f"Fix:\n```bash\n{report.hint}\n```"
            )
            await msg.update()
            logger.warning("health_check failed: %s", report.reason)
            return
        cl.user_session.set("ollama_ok", True)

    try:
        for delta in stream_chat(
            message.content,
            images=images,
            history=[sys_msg],
            num_ctx=28_000,
        ):
            if first_token_t is None:
                first_token_t = time.time()
            raw += delta
            await msg.stream_token(delta)
    except Exception as e:
        # Daemon went away mid-stream, OOM, or context overflow.
        logger.exception("stream_chat failed")
        msg.content = (
            f"**Streaming failed mid-answer.**\n\n"
            f"```\n{type(e).__name__}: {e}\n```\n\n"
            "This usually means the Ollama daemon crashed or ran out of "
            "VRAM. Check `ollama logs` and try again."
        )
        await msg.update()
        return

    # Replace the streamed raw text with a formatted version that has
    # numbered footnotes + a citation appendix.
    body, cites = _format_answer(raw, known_cite_ids)
    appendix = _citation_appendix(cites)

    elapsed = time.time() - t0
    gen_t = max(0.01, time.time() - (first_token_t or t0))
    tokens = max(1, len(raw) // 4)
    tps = tokens / gen_t
    foot = (
        f"\n\n_Answered in {elapsed:.1f}s · ~{tps:.0f} tok/s · "
        f"{len(cites)} citation{'s' if len(cites) != 1 else ''}_"
    )
    msg.content = body + appendix + foot
    await msg.update()
