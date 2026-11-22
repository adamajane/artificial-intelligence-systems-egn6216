"""Screenshots of the dashboard pages (plan 6.5 HR4: a greyscale screenshot of each page; walkthrough and test evidence).

  uv run python deploy/tools/screens.py --url http://127.0.0.1:8501 --out docs/figures/m5/screens [--grey] [--query "t=...&scenario=stale"]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {"overview": "", "region": "region", "tracking": "tracking", "about": "about"}
GREY = "html { filter: grayscale(100%) !important; }"


def settle(page, timeout_s: float = 30) -> float:
    """Wait until Streamlit finished running the script; returns seconds waited."""
    t0 = time.perf_counter()
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout_s * 1e3)
    page.wait_for_selector("[data-ss-ready]", state="attached", timeout=timeout_s * 1e3)
    end = time.perf_counter() + timeout_s
    quiet = 0
    while time.perf_counter() < end:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        quiet = quiet + 1 if not running else 0
        if quiet >= 3:
            break
        time.sleep(0.1)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--pages", nargs="+", default=list(PAGES))
    ap.add_argument("--grey", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--width", type=int, default=1440)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": args.width, "height": 1000})
        for name in args.pages:
            url = f"{args.url.rstrip('/')}/{PAGES[name]}" + (f"?{args.query}" if args.query else "")
            page.goto(url)
            settle(page)
            page.wait_for_selector(f'[data-ss-ready^="{name}|"]', state="attached", timeout=30e3)
            if args.grey:
                page.add_style_tag(content=GREY)
                time.sleep(0.3)
            f = args.out / f"{name}{('_' + args.tag) if args.tag else ''}{'_grey' if args.grey else ''}.png"
            page.screenshot(path=str(f), full_page=True)
            print(f)
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
