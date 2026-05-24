"""End-to-end smoke test: ingest a synthetic multi-page PDF, ask a
question grounded in one specific page, confirm the model cites it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import fitz

from deepread.citations import citation_prompt, extract_citations
from deepread.ingest import ingest_pdf
from deepread.llm import stream_chat


def make_synth_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Page 1 - Introduction. We study cats.", fontsize=24)
    doc.new_page().insert_text((72, 72), "Page 2 - Methods. Subjects fed twice daily.", fontsize=24)
    doc.new_page().insert_text((72, 72), "Page 3 - Results. Mean weight was 4.7 kg.", fontsize=24)
    doc.new_page().insert_text((72, 72), "Page 4 - Discussion. Cats prefer salmon.", fontsize=24)
    doc.save(path)
    doc.close()


def main() -> int:
    pdf = Path("/tmp/deepread_smoke.pdf")
    make_synth_pdf(pdf)
    shards = ingest_pdf(pdf)
    print(f"Ingested {len(shards)} shards from {pdf.name}")

    sys_prompt = citation_prompt(shards)
    history = [{"role": "system", "content": sys_prompt}]
    images = [s.png_bytes for s in shards]

    question = "What was the mean weight reported? Cite the page."
    print(f"\n>>> Q: {question}\n")

    t0 = time.time()
    answer_parts: list[str] = []
    for delta in stream_chat(
        question, images=images, history=history, num_ctx=12_000
    ):
        sys.stdout.write(delta)
        sys.stdout.flush()
        answer_parts.append(delta)
    elapsed = time.time() - t0
    answer = "".join(answer_parts)

    print(f"\n--- {elapsed:.1f}s ---")
    cites = extract_citations(answer)
    print(f"Citations extracted: {cites}")

    cited_p3 = any("p3" in c for c in cites)
    mentions_47 = "4.7" in answer
    print(f"Cited page 3: {cited_p3}")
    print(f"Mentions 4.7 kg: {mentions_47}")
    return 0 if (cited_p3 and mentions_47) else 1


if __name__ == "__main__":
    sys.exit(main())
