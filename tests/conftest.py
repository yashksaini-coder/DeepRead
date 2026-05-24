"""Shared pytest fixtures.

The `synthetic_pdf` fixture builds a 3-page PDF in memory so
`test_ingest.py` doesn't depend on a checked-in binary.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import pytest


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """A 3-page PDF with one short paragraph per page."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"This is page {i + 1} of the test document.")
    out = tmp_path / "synth.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def synthetic_png(tmp_path: Path) -> Path:
    from PIL import Image

    img = Image.new("RGB", (320, 240), color=(220, 220, 220))
    out = tmp_path / "chart.png"
    img.save(out, format="PNG")
    return out
