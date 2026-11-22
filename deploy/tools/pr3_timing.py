"""PR3: 100 dashboard view changes in a real browser, p95 <= 2 s (plan 10).

Alternates "6 h later" on the Overview with opening Region detail and going back. Each change is timed from the click
until the page's ready marker (``data-ss-ready``, written at the end of the Streamlit run) shows the new state.

  uv run python deploy/tools/pr3_timing.py --url http://127.0.0.1:8501 --n 100 --out experiments/results/m5/tests/pr3_timing.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

READY = "[data-ss-ready]"


def marker(page) -> str:
    return page.locator(READY).last.get_attribute("data-ss-ready") or ""


def wait_change(page, before: str, prefix: str, timeout_s: float = 30) -> None:
    page.wait_for_function(
        "([b, p]) => { const m = [...document.querySelectorAll('[data-ss-ready]')].pop(); "
        "return m && m.dataset.ssReady !== b && m.dataset.ssReady.startsWith(p) && !document.querySelector('[data-stale=\"true\"]'); }",
        arg=[before, prefix], timeout=timeout_s * 1e3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--query", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    times, kinds = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.url + (f"/?{args.query}" if args.query else "/"))
        page.wait_for_selector(READY, state="attached", timeout=60e3)
        for i in range(args.n):
            before = marker(page)
            if i % 4 in (0, 1):  # next date (twice), then region detail and back
                t0 = time.perf_counter()
                page.get_by_role("button", name="6 h later ▶").click()
                wait_change(page, before, "overview|")
                kinds.append("next_date")
            elif i % 4 == 2:
                t0 = time.perf_counter()
                page.get_by_role("link", name="Region detail").click()
                wait_change(page, before, "region|")
                kinds.append("open_region")
            else:
                t0 = time.perf_counter()
                page.get_by_role("link", name="Overview").click()
                wait_change(page, before, "overview|")
                kinds.append("back_to_overview")
            times.append(time.perf_counter() - t0)
        b.close()
    s = sorted(times)
    res = {"measured_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "url": args.url, "n": len(times),
           "p50_s": statistics.median(s), "p95_s": s[int(0.95 * (len(s) - 1))], "max_s": s[-1],
           "share_over_2s": sum(x > 2 for x in s) / len(s), "target_p95_s": 2.0, "pass": s[int(0.95 * (len(s) - 1))] <= 2.0,
           "by_kind": {k: {"n": kinds.count(k), "p50_s": statistics.median([t for t, kk in zip(times, kinds) if kk == k])} for k in set(kinds)},
           "times_s": [round(x, 4) for x in times], "kinds": kinds}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("times_s", "kinds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
