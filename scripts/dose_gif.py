"""Animated GIF of one neuron's dose sweep in the viewer (README "video").

Local explanation: serves site/ locally, opens a pole page in headless Chromium (Playwright), clicks each dose button
of the "Turn it up" panel and screenshots the steering section after each click; Pillow assembles the frames into a
looping GIF (https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving).
Usage: uv run python scripts/dose_gif.py "neuron:L30:N2121:+" docs/images/dose_sweep.gif
"""

import functools
import http.server
import io
import sys
import threading
import urllib.parse

from PIL import Image
from playwright.sync_api import sync_playwright


def main(key: str, out: str, port: int = 8766) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="site")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    frames = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/index.html?key={urllib.parse.quote(key)}")
            page.wait_for_selector("#levels button")
            buttons = page.locator("#levels button")
            for n in range(buttons.count()):
                buttons.nth(n).click()
                page.wait_for_timeout(150)
                box = page.locator("#levels").bounding_box()  # crop: dose buttons + the generations below them
                clip = {"x": 0, "y": max(0, box["y"] - 330), "width": 1100, "height": 760}
                frames.append(Image.open(io.BytesIO(page.screenshot(clip=clip, full_page=True))).convert("P", palette=Image.ADAPTIVE))
            browser.close()
    finally:
        server.shutdown()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1600, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
