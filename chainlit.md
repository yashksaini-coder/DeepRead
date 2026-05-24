# DeepRead

**Local document Q&A.** Drop in a PDF, ask anything about it, get an answer with page-anchored citations.

Everything runs on this machine. The model is `gemma4:e4b` (Google's Gemma 4 E4B with a 128K context window and native vision) served by Ollama. No cloud calls, no per-query cost, no telemetry.

## How to use

1. **Pick a paper** from the right sidebar — five classics are bundled (Attention, GFS, MapReduce, Raft, Bitcoin), or upload your own PDF or image.
2. **Ask anything** — the question goes to Gemma 4 along with the rendered page images. Answers stream back word-by-word.
3. **Read the citations** — each `[^1]`-style footnote resolves to the specific page the model answered from. The model is constrained to cite only ids that correspond to real pages, so hallucinated page numbers can't happen.

The context-budget chart in the right sidebar is color-coded by tier — green under 20K tokens (interactive), amber from 60K (research/batch), red past the safe limit. Loading more papers updates it live. Click the **Context** button (top-right of the header) to reopen the sidebar if you closed it.

## Benchmark diagnostics

Type `/bench` commands in the same chat — no profile switch, no chat clearing:

- `/bench show` — render the latest context-window sweep results as interactive Plotly charts
- `/bench run --ctx 5000 20000 60000 --needles 5` — kick off a fresh sweep against the local Ollama. Several minutes at large context sizes (TTFT grows ~linearly to ~72s at 100K).
