"""Build reports/paper.pdf from reports/paper.md: pandoc (Markdown -> standalone HTML) + headless Chromium (HTML -> PDF).

Local explanation: pypandoc wraps the pandoc binary (https://pandoc.org/MANUAL.html); `--embed-resources` inlines
figures so the HTML is self-contained, `--mathml` renders formulas natively in Chromium. Playwright's `page.pdf`
(https://playwright.dev/python/docs/api/class-page#page-pdf) prints the page with A4 margins.
Usage: uv run python scripts/build_paper.py
Global context: avoids a LaTeX installation; the same Chromium is used for README screenshots.
"""

from pathlib import Path

import pypandoc
from playwright.sync_api import sync_playwright

CSS = """
body { font-family: 'Times New Roman', Georgia, serif; font-size: 11pt; line-height: 1.45; max-width: 46em; margin: auto; color: #111; }
h1 { font-size: 18pt; text-align: center; margin-bottom: 0.2em; } h2 { font-size: 13pt; margin-top: 1.4em; } h3 { font-size: 11.5pt; }
p.author, p.date { text-align: center; margin: 0.2em; }
table { border-collapse: collapse; font-size: 9pt; margin: 0.8em auto; } th, td { border-bottom: 1px solid #bbb; padding: 3px 6px; text-align: left; }
img { max-width: 100%; display: block; margin: 0.6em auto; } figure { margin: 0.8em 0; } figcaption { font-size: 9pt; text-align: center; }
code { font-size: 9pt; } pre { font-size: 8.5pt; background: #f5f5f5; padding: 6px; white-space: pre-wrap; }
a { color: #1a4f9c; text-decoration: none; }
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md, html, pdf = root / "reports/paper.md", root / "reports/paper.html", root / "reports/paper.pdf"
    css = root / "reports/paper.css"
    css.write_text(CSS, encoding="utf-8")
    pypandoc.convert_file(str(md), "html5", outputfile=str(html),
                          extra_args=["--standalone", "--embed-resources", f"--css={css}", "--mathml",
                                      f"--resource-path={root / 'reports'}", "--metadata", "lang=en"])
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html.as_uri())
        page.pdf(path=str(pdf), format="A4", margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                 print_background=True)
        browser.close()
    html.unlink()
    css.unlink()
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
