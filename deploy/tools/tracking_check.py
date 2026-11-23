"""Plan 10 integration check: Tracking figures on the dashboard vs a recomputation from the store, at fixed dates.

  uv run python deploy/tools/tracking_check.py --store data/store/stack/solarsentinel.db --run m5_final_test_0101774eb973
"""

import argparse
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8501")
ap.add_argument("--store", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--dates", nargs="+", default=["2016-07-01T00:00:00Z", "2017-09-05T12:00:00Z", "2017-12-29T00:00:00Z"])
ap.add_argument("--out", default="experiments/results/m5/tests/tracking_dashboard_vs_store.json")
a = ap.parse_args()
con = sqlite3.connect(a.store)
thr = con.execute("SELECT threshold FROM runs WHERE run_id = ?", (a.run,)).fetchone()[0]
rows = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    for d in a.dates:
        known = (pd.Timestamp(d) - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        x = pd.read_sql("SELECT p.p, o.label FROM predictions p JOIN outcomes o ON o.run_id = p.run_id AND o.harpnum = p.harpnum AND "
                        "o.t_slot = p.t_slot WHERE p.run_id = ? AND p.t_slot <= ?", con, params=(a.run, known))
        y, al = x.label.to_numpy() == 1, x.p.to_numpy() >= thr
        tp, fp, fn, tn = int((y & al).sum()), int((~y & al).sum()), int((y & ~al).sum()), int((~y & ~al).sum())
        store = {"caught": f"{tp} of {tp + fn}", "hit_share": f"{tp / (tp + fp):.0%}" if tp + fp else "no alerts",
                 "tss": f"{tp / max(tp + fn, 1) - fp / max(fp + tn, 1):.2f}", "brier": f"{float(np.mean((x.p - y) ** 2)):.4f}"}
        pg.goto(f"{a.url}/tracking?run={a.run}&t={d}")
        pg.wait_for_selector('[data-ss-ready^="tracking|"]', state="attached", timeout=60e3)
        time.sleep(0.5)
        vals = pg.locator('[data-testid="stMetricValue"]').all_inner_texts()
        dash = dict(zip(["caught", "hit_share", "tss", "brier"], vals))
        rows.append({"date": d, "known_up_to": known, "n_known": len(x), "store": store, "dashboard": dash, "match": dash == store})
    b.close()
out = {"run": a.run, "threshold": thr, "rows": rows, "all_match": all(r["match"] for r in rows)}
open(a.out, "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
