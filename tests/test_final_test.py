"""Final-test evaluation (docs/4-milestone5-preregistration.md): label gate, store provenance, dev reproduction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from solarsentinel.config import repo_root
from solarsentinel.models import dataset
from solarsentinel.models import final_test as ftm
from solarsentinel.serve import store

RESULTS = repo_root() / "experiments/results/m5"
SCORES = RESULTS / "m5_bundle_resnet18_pad_s42_val_scores.parquet"
REPORT = RESULTS / "m5_bundle_resnet18_pad_s42_report.json"
needs_bundle = pytest.mark.skipif(not SCORES.exists(), reason="bundle val scores not built (models.bundle)")


def test_gate_refuses_uncommitted_preregistration(monkeypatch):
    monkeypatch.setattr(ftm, "PREREG", "docs/does-not-exist-preregistration.md")
    with pytest.raises(PermissionError):
        ftm.check_preregistration()
    with pytest.raises(PermissionError):
        ftm.replay_outcomes("test: must be refused")


def test_replay_needs_a_reason():
    with pytest.raises(ValueError):
        ftm.observations_for("replay", None)


def test_null_provenance_counted(tmp_path):
    con = store.connect(tmp_path / "s.db")
    store.start_run(con, "r", "test", "unit test", bundle_id="b", threshold=0.05, mode="probability", args={})
    row = dict(run_id="r", harpnum=1, noaa_ars="1", t_slot="2015-01-01T00:00:00Z", t_rec="t", p=0.1, p_raw=0.2, alert=1,
               mode="probability", bundle_id="b", threshold=0.05, source="test", quality=0, request_id="x", latency_ms=1.0,
               scored_at=store.now())
    store.insert_rows(con, "predictions", [row, row | {"harpnum": 2, "request_id": ""}])
    con.commit()
    assert store.null_provenance(con, "r") == 1
    with pytest.raises(ValueError):
        store.start_run(con, "r", "test", "again", bundle_id="b", threshold=0.05, mode="probability", args={})


def test_feedback_note_length_limited(tmp_path):
    import sqlite3

    con = store.connect(tmp_path / "s.db")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO feedback (run_id, harpnum, t_slot, note, created_at) VALUES ('r', 1, 't', ?, 'now')", ("x" * 501,))


@needs_bundle
def test_dev_stream_reproduces_bundle_val(tmp_path):
    s = pd.read_parquet(SCORES)
    rep = json.loads(REPORT.read_text())
    thr, bid = rep["threshold"]["calibrated"], rep["bundle"]["id"]
    obs = dataset.load_observations("v2", ("val",)).astype({"T_REC": str})
    s = s.merge(obs[["HARPNUM", "T_REC", "slot_time", "NOAA_ARS", "QUALITY"]], on=["HARPNUM", "T_REC"])
    path = tmp_path / "dev.db"
    con = store.connect(path)
    store.start_run(con, "dev", "dev", "unit test", bundle_id=bid, threshold=thr, mode="probability", args={})
    store.insert_rows(con, "predictions", [dict(
        run_id="dev", harpnum=int(r.HARPNUM), noaa_ars=r.NOAA_ARS, t_slot=store.iso(r.slot_time), t_rec=r.T_REC, p=float(r.p_cal),
        p_raw=float(r.p_raw_cpu), alert=int(r.p_cal >= thr), mode="probability", bundle_id=bid, threshold=thr, source="dev",
        quality=int(r.QUALITY), request_id=f"r{i}", latency_ms=1.0, scored_at=store.now()) for i, r in enumerate(s.itertuples())])
    con.commit()
    df, integ = ftm.frame(path, "dev", "val")
    res = ftm.evaluate(df, "val", integ["period_days"])
    a = res["subsets"]["all_records"]
    assert integ["null_provenance_fields"] == 0
    assert a["pr1"]["n"] == 626 and a["pr1"]["n_pos"] == 208
    assert a["pr1"]["model"]["tss"] == pytest.approx(rep["val_metrics_calibrated"]["tss"], abs=1e-12)
    assert a["pr2"]["pass"]
    assert res["operator_view"]["all_records"]["alerts_per_day"] == pytest.approx(rep["operator_view_2015"]["alerts_per_day"], abs=0.01)
    low = next(r for r in res["slices"]["all_records"] if r["slice"] == "usflux_tercile" and r["level"] == "low")
    assert low["recall"] == pytest.approx(0.557, abs=0.001)  # M4 work report 10.6
    assert np.isclose(res["threshold"], thr)
