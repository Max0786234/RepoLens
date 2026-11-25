"""Record docs/demo.gif from the running Streamlit app (dev-only helper).

    pip install playwright
    REPOLENS_PUBLIC_DEMO=1 REPOLENS_PROVIDER=groq REPOLENS_EMBED_BACKEND=none \
        streamlit run app.py --server.port 8599 --server.headless true
    python scripts/record_demo.py --url http://localhost:8599

Takes screenshots while a real question is answered and stitches them into an
animated GIF with Pillow (installed with Streamlit). Uses the system Edge/Chrome.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"
WIDTH, HEIGHT = 1280, 760
HIDE_TOOLBAR = '[data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}'


class Recorder:
    def __init__(self, page, scale: float) -> None:
        self.page = page
        self.scale = scale
        self.frames: list[tuple[Image.Image, int]] = []

    def snap(self, hold_ms: int = 120) -> None:
        png = self.page.screenshot()
        image = Image.open(io.BytesIO(png)).convert("RGB")
        if self.scale != 1:
            image = image.resize((int(image.width * self.scale), int(image.height * self.scale)), Image.LANCZOS)
        self.frames.append((image, hold_ms))

    def hold(self, ms: int) -> None:
        if self.frames:
            image, duration = self.frames[-1]
            self.frames[-1] = (image, duration + ms)

    def record_until(self, done, timeout_s: float, every_ms: int = 450) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.snap(every_ms)
            if done():
                return
            self.page.wait_for_timeout(every_ms)
        raise TimeoutError("demo step did not finish in time")

    def save(self, path: Path) -> None:
        images = [frame for frame, _ in self.frames]
        durations = [duration for _, duration in self.frames]
        palette = [
            img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for img in images
        ]
        palette[0].save(path, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)


def compress_gif(path: Path, max_hold_ms: int = 600, final_hold_ms: int = 4_000, threshold: float = 0.02) -> None:
    """Time-lapse a recording: merge near-identical frames and cap how long any frame is held.

    Waiting on a rate-limited API produces long runs of identical "Thinking..." frames;
    this keeps the story (typing, live trace, answer) and drops the dead time.
    """
    from PIL import ImageChops, ImageStat

    source = Image.open(path)
    frames: list[tuple[Image.Image, int]] = []
    for i in range(source.n_frames):
        source.seek(i)
        frame = source.convert("RGB")
        duration = source.info.get("duration", 100)
        if frames:
            diff = ImageStat.Stat(ImageChops.difference(frame, frames[-1][0])).mean
            if sum(diff) / 3 < threshold:
                frames[-1] = (frames[-1][0], frames[-1][1] + duration)
                continue
        frames.append((frame, duration))

    def shorten(duration: int) -> int:
        # Frames captured while polling a slow step (record_until, ~450-700 ms apart) are fast-forwarded.
        if 400 <= duration <= 1_000:
            return 200
        return min(duration, max_hold_ms)

    capped = [(frames[0][0], 2_000)] + [(img, shorten(d)) for img, d in frames[1:-1]]
    capped.append((frames[-1][0], final_hold_ms))
    images = [img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for img, _ in capped]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=[d for _, d in capped], loop=0, optimize=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    parser.add_argument("--sample", default="psf/requests", help="Label of the sample-repo button")
    parser.add_argument("--question", default="Where are connection retries configured, and what is the default?")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--scale", type=float, default=0.75)
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        rec = Recorder(page, args.scale)
        page.goto(args.url)
        page.get_by_text("Ask questions about any codebase").wait_for(timeout=60_000)
        page.add_style_tag(content=HIDE_TOOLBAR)
        page.wait_for_timeout(800)
        rec.snap(1800)

        page.get_by_role("tab", name="Try a sample").click()
        page.wait_for_timeout(500)
        rec.snap(900)
        page.get_by_role("button", name=args.sample).click()
        rec.record_until(lambda: page.get_by_text("Try asking:").count() > 0, timeout_s=600, every_ms=700)
        page.add_style_tag(content=HIDE_TOOLBAR)
        rec.hold(1200)

        box = page.get_by_placeholder("Ask about this codebase")
        box.click()
        for i in range(0, len(args.question), 6):
            box.type(args.question[i : i + 6], delay=15)
            rec.snap(90)
        rec.hold(500)
        box.press("Enter")
        rec.record_until(lambda: page.get_by_text("citations verified").count() > 0, timeout_s=300)
        page.wait_for_timeout(800)
        rec.snap(2500)

        # Open the citation panel and scroll through the answer.
        citations = page.get_by_text("Citations", exact=True)
        if citations.count():
            citations.last.click()
            page.wait_for_timeout(600)
        page.locator('[data-testid="stChatMessage"]').last.scroll_into_view_if_needed()
        rec.snap(3500)

        browser.close()
    OUT.parent.mkdir(exist_ok=True)
    rec.save(OUT)
    compress_gif(OUT)
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(rec.frames)} frames)")


if __name__ == "__main__":
    main()
