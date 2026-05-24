"""Vision smoke test: ask Gemma 4 E4B to describe a generated image.

We generate a small image with a known caption baked into it so the
test is reproducible without an internet download.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from deepread.llm import stream_chat


def make_test_image() -> bytes:
    img = Image.new("RGB", (640, 240), color="white")
    draw = ImageDraw.Draw(img)
    text = "DeepRead vision smoke test\nThe word HELLO is here."
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 70), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    print(">>> gemma4:e4b vision smoke test")
    png = make_test_image()
    Path("/tmp/deepread_vision_test.png").write_bytes(png)

    t0 = time.time()
    out: list[str] = []
    for delta in stream_chat(
        "What words appear in this image? Answer in one short sentence.",
        images=[png],
        num_ctx=8000,
    ):
        sys.stdout.write(delta)
        sys.stdout.flush()
        out.append(delta)
    elapsed = time.time() - t0
    text = "".join(out)
    print(f"\n--- {elapsed:.1f}s, {len(text)} chars ---")

    ok = "HELLO" in text.upper() or "deepread" in text.lower()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
