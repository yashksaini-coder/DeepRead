from __future__ import annotations

from deepread.budget import (
    CONTEXT_LIMIT,
    SAFE_LIMIT,
    TOKENS_PER_IMAGE,
    budget,
    estimate,
)
from deepread.ingest import Shard


def _img_shard(name: str = "doc.pdf#p1") -> Shard:
    return Shard(cite_id=name, source="doc.pdf", page=1, png_bytes=b"\x89PNG..", extracted_text="")


def _text_shard(text: str, name: str = "notes.md") -> Shard:
    return Shard(cite_id=name, source=name, page=None, png_bytes=b"", extracted_text=text)


def test_empty_shard_list_uses_zero_tokens():
    b = budget([])
    assert b["used"] == 0
    assert b["pct"] == 0
    assert not b["over_safe"]
    assert not b["over_limit"]


def test_image_shard_costs_fixed_amount():
    assert estimate(_img_shard()) == TOKENS_PER_IMAGE


def test_pct_monotonically_increases_with_shards():
    b1 = budget([_img_shard("a#p1")])
    b2 = budget([_img_shard("a#p1"), _img_shard("a#p2")])
    b3 = budget([_img_shard(f"a#p{i}") for i in range(10)])
    assert b1["pct"] < b2["pct"] < b3["pct"]


def test_over_safe_flips_at_threshold():
    # Enough image shards to cross SAFE_LIMIT but stay under CONTEXT_LIMIT
    n = (SAFE_LIMIT // TOKENS_PER_IMAGE) + 2
    shards = [_img_shard(f"p{i}") for i in range(n)]
    b = budget(shards)
    assert b["used"] > SAFE_LIMIT
    assert b["used"] <= CONTEXT_LIMIT
    assert b["over_safe"]
    assert not b["over_limit"]


def test_over_limit_flips_when_truly_over():
    n = (CONTEXT_LIMIT // TOKENS_PER_IMAGE) + 2
    shards = [_img_shard(f"p{i}") for i in range(n)]
    b = budget(shards)
    assert b["over_limit"]
    assert b["pct"] == 100  # capped at 100


def test_text_shard_token_count_scales_with_length():
    short = _text_shard("x" * 100)
    long = _text_shard("x" * 10_000)
    assert estimate(long) > estimate(short)
