"""PDF and image ingestion → uniform `Shard` objects.

A `Shard` is the smallest renderable unit the model will see, paired
with a stable `cite_id` the model can echo back in citations.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image


@dataclass(slots=True, frozen=True)
class Shard:
    cite_id: str
    source: str
    page: int | None
    png_bytes: bytes
    extracted_text: str


def ingest_pdf(path: str | Path, dpi: int = 150) -> list[Shard]:
    """Render every page of a PDF to PNG and extract its text.

    DPI is a tradeoff: 150 is the sweet spot — readable charts and
    small-print footnotes survive, but each page stays under ~250 KB
    so the model's vision encoder isn't asked to do unreasonable work.
    """
    path = Path(path)
    doc = fitz.open(path)
    name = path.name
    shards: list[Shard] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        shards.append(
            Shard(
                cite_id=f"{name}#p{i + 1}",
                source=name,
                page=i + 1,
                png_bytes=buf.getvalue(),
                extracted_text=page.get_text() or "",
            )
        )
    doc.close()
    return shards


def ingest_image(path: str | Path) -> Shard:
    path = Path(path)
    return Shard(
        cite_id=path.name,
        source=path.name,
        page=None,
        png_bytes=path.read_bytes(),
        extracted_text="",
    )
