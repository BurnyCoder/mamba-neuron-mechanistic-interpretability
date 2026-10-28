"""Render README images from the static viewer (site/) with headless Chromium (Playwright).

Local explanation: serves site/ on a local port with the stdlib http.server
(https://docs.python.org/3/library/http.server.html), opens `index.html?key=<pole>` in Playwright's sync API
(https://playwright.dev/python/docs/screenshots) and saves a full-page PNG.
Usage: uv run python scripts/screenshot_viewer.py "neuron:L37:N1812:+" docs/images/viewer.png
Global context: the README hero image is a screenshot of the same viewer that is deployed to GitHub Pages.
"""

import functools
import http.server
import sys
import threading
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(key: str, out: str, port: int = 8765) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="site")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5)
            page.goto(f"http://127.0.0.1:{port}/index.html?key={urllib.parse.quote(key)}")
            page.wait_for_selector("#detail h2")  # pole detail rendered
            page.wait_for_timeout(500)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=out, full_page=False)
            browser.close()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
