from __future__ import annotations

from deepread.ingest import ingest_pdf, ingest_image


def test_ingest_pdf_produces_one_shard_per_page(synthetic_pdf):
    shards = ingest_pdf(synthetic_pdf)
    assert len(shards) == 3


def test_ingest_pdf_cite_ids_are_unique_and_well_formed(synthetic_pdf):
    shards = ingest_pdf(synthetic_pdf)
    cite_ids = [s.cite_id for s in shards]
    assert len(cite_ids) == len(set(cite_ids))
    assert cite_ids == [f"{synthetic_pdf.name}#p{i}" for i in (1, 2, 3)]


def test_ingest_pdf_pages_have_png_bytes_and_text(synthetic_pdf):
    shards = ingest_pdf(synthetic_pdf)
    for s in shards:
        assert s.png_bytes.startswith(b"\x89PNG"), "PNG magic bytes expected"
        assert "test document" in s.extracted_text
        assert s.page is not None


def test_ingest_image_yields_single_shard_with_filename_cite_id(synthetic_png):
    shard = ingest_image(synthetic_png)
    assert shard.cite_id == synthetic_png.name
    assert shard.page is None
    assert shard.png_bytes.startswith(b"\x89PNG")
