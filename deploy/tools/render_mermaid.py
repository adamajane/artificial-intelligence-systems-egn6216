"""Render a mermaid file to PNG with headless Chromium (Playwright) and mermaid 11.4.1 from jsDelivr.

  uv run python deploy/tools/render_mermaid.py docs/figures/m5/architecture_v2.mmd docs/figures/m5/architecture_v2.png
"""

import html
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

src, out = Path(sys.argv[1]), Path(sys.argv[2])
page_html = f"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script></head>
<body style="background:#fff;margin:0;padding:16px"><pre class="mermaid">{html.escape(src.read_text())}</pre>
<script>mermaid.initialize({{startOnLoad:false, theme:'base', flowchart:{{htmlLabels:true, curve:'basis'}},
 themeVariables:{{fontFamily:'Helvetica, Arial, sans-serif', fontSize:'14px'}}}});
mermaid.run().then(()=>{{document.body.dataset.done='1'}});</script></body></html>"""
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 2200, "height": 1400}, device_scale_factor=2)
    pg.set_content(page_html)
    pg.wait_for_selector("body[data-done='1']", timeout=60000)
    pg.locator("svg").first.screenshot(path=str(out))
    b.close()
print(out)
