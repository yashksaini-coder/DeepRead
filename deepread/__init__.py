"""DeepRead — local document intelligence powered by Gemma 4 E4B.

Public surface kept tiny on purpose: the building blocks are composed
in `app.py` (UI) and `benchmarks/run_context_sweep.py` (benchmarks).
"""

from .ingest import Shard, ingest_pdf, ingest_image
from .budget import budget, estimate, CONTEXT_LIMIT, SAFE_LIMIT
from .citations import citation_prompt, extract_citations
from .llm import stream_chat, MODEL
from .papers import EXAMPLE_PAPERS, ExamplePaper, download as download_paper, get as get_paper

__all__ = [
    "Shard",
    "ingest_pdf",
    "ingest_image",
    "budget",
    "estimate",
    "CONTEXT_LIMIT",
    "SAFE_LIMIT",
    "citation_prompt",
    "extract_citations",
    "stream_chat",
    "MODEL",
    "EXAMPLE_PAPERS",
    "ExamplePaper",
    "download_paper",
    "get_paper",
]
