"""Source-grounded citations.

We don't post-process the model's answer to find page numbers — we
*pre-condition* it to emit `[[cite_id]]` tokens by giving it a fixed
citation catalog. This keeps citations honest: the model can only
choose from ids that correspond to material actually in the prompt.

In practice the model occasionally emits single brackets (`[id]`)
instead of double (`[[id]]`). We accept either form — the regex
prefers double brackets but falls back to single brackets when the
contents look like a cite_id (contains `#` or `.` and no spaces).
"""

from __future__ import annotations

import re

from .ingest import Shard

# Match double-bracketed citations OR single-bracketed citations whose
# contents look like a cite_id (has #p_ or matches name.ext patterns
# without spaces). We deliberately keep this conservative so we don't
# eat plain markdown links like [text](url).
CITATION_RE = re.compile(
    r"\[\[(?P<id>[^\]\n]+?)\]\]"      # preferred: [[id]]
    r"|\[(?P<idSingle>[A-Za-z0-9_\-.]+(?:\.[A-Za-z0-9]+)?(?:#p\d+)?)\]",
)


def citation_prompt(shards: list[Shard]) -> str:
    if not shards:
        return (
            "You are a careful research assistant. No documents have been "
            "loaded yet, so politely ask the user to add a PDF or image."
        )
    catalog = "\n".join(f"- {s.cite_id}" for s in shards)
    return (
        "You are a careful research assistant. The user has loaded the "
        "documents and images listed below. When you make a factual claim "
        "drawn from a specific page or image, cite it inline using the "
        "exact format [[cite_id]] — that's TWO opening square brackets "
        "and TWO closing square brackets around the id. Use ONLY these "
        "citation ids:\n"
        f"{catalog}\n\n"
        "Example: \"The model uses sliding-window attention [[paper.pdf#p4]].\"\n"
        "If a claim isn't supported by the loaded material, say so plainly. "
        "Do not invent page numbers. Do not cite ids not in the list above. "
        "Use double brackets, never single."
    )


def extract_citations(answer: str, known: set[str] | None = None) -> list[str]:
    """Return citations in first-seen order, deduplicated.

    `known` optionally restricts matches to a set of valid cite_ids.
    Single-bracketed candidates are only accepted if they're in the
    `known` set — this prevents grabbing things like `[1]` or `[note]`.
    """
    seen: dict[str, None] = {}
    for m in CITATION_RE.finditer(answer):
        cid = (m.group("id") or m.group("idSingle") or "").strip()
        if not cid:
            continue
        if m.group("idSingle") is not None and known is not None and cid not in known:
            continue
        seen[cid] = None
    return list(seen.keys())
