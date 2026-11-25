"""Capture README screenshots of the running Streamlit app.

Dev-only helper (Playwright is not a project dependency):

    pip install playwright
    streamlit run app.py --server.port 8599 --server.headless true
    python scripts/capture_screenshots.py --url http://localhost:8599 --repo https://github.com/psf/requests

Uses the locally installed Microsoft Edge or Chrome (``--channel``), so no browser
download is needed. Answers on CPU-only machines can take minutes; the timeout
is generous.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    parser.add_argument("--repo", default="https://github.com/psf/requests")
    parser.add_argument(
        "--question", default="How does requests decide whether to drop the Authorization header on a redirect?"
    )
    parser.add_argument("--mode", default="Agent", help="Label prefix of the answer-mode radio option")
    parser.add_argument("--channel", default="msedge", help="msedge or chrome")
    parser.add_argument("--timeout-min", type=float, default=20)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(args.url)
        page.get_by_text("Ask questions about any codebase").wait_for(timeout=60_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "landing.png")
        print("saved landing.png")

        page.get_by_placeholder("https://github.com/psf/requests").fill(args.repo)
        page.keyboard.press("Enter")
        page.get_by_role("button", name="Index repository").click()
        page.get_by_text("Try asking:").wait_for(timeout=15 * 60_000)

        page.get_by_text(args.mode, exact=False).first.click()
        page.wait_for_timeout(500)
        box = page.get_by_placeholder("Ask about this codebase")
        box.fill(args.question)
        box.press("Enter")
        page.get_by_text("citations verified").first.wait_for(timeout=int(args.timeout_min * 60_000))
        page.wait_for_timeout(1500)
        for label in ("Citations", "Investigation trace"):
            expander = page.get_by_text(label, exact=True)
            if expander.count():
                expander.first.click()
                page.wait_for_timeout(400)
        page.screenshot(path=OUT / "answer.png", full_page=True)
        print("saved answer.png")
        browser.close()


if __name__ == "__main__":
    main()
