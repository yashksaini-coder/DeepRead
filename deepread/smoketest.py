"""Single-command smoketest of the full pipeline.

Run: `uv run python -m deepread.smoketest`

Exits 0 on success, non-zero on failure. Used in CI and as a quick
"does my install actually work" check after `uv sync`.
"""

from __future__ import annotations

import sys

from .citations import citation_prompt, extract_citations
from .ingest import Shard


def main() -> int:
    # In-memory pipeline check — no network, no Ollama call.
    shards = [
        Shard(cite_id="a.pdf#p1", source="a.pdf", page=1, png_bytes=b"\x89PNG..", extracted_text=""),
        Shard(cite_id="b.png", source="b.png", page=None, png_bytes=b"\x89PNG..", extracted_text=""),
    ]
    prompt = citation_prompt(shards)
    assert "a.pdf#p1" in prompt
    assert "b.png" in prompt
    found = extract_citations("See [[a.pdf#p1]] and [[b.png]].")
    assert found == ["a.pdf#p1", "b.png"], found
    print("deepread smoketest: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
