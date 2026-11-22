"""Write deploy/grafana/dashboards/solarsentinel.json from the monitoring config (thresholds drawn as lines)."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
mon = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())["m5"]["monitoring"]
DS = {"type": "prometheus", "uid": "prometheus"}


def ts(title, expr, x, y, lines=(), unit="none", legend="{{run_id}}", w=8, h=7, desc=""):
    steps = [{"color": "green", "value": None}] + [{"color": "red", "value": v} for v in lines]
    limits = {"min": 0, "max": 1} if unit == "percentunit" else {}
    return {"type": "timeseries", "title": title, "description": desc, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"refId": "A", "expr": expr, "legendFormat": legend, "datasource": DS}],
            "fieldConfig": {"defaults": {"unit": unit, **limits, "custom": {"thresholdsStyle": {"mode": "dashed" if lines else "off"},
                                                                    "spanNulls": True, "lineWidth": 2, "showPoints": "always", "pointSize": 4},
                                         "color": {"mode": "fixed", "fixedColor": "#5794F2"},
                                         "thresholds": {"mode": "absolute", "steps": steps}}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}


def sig(name):
    return f'solarsentinel_signal{{signal="{name}"}}'


s3 = mon["s3_alert_share"]
panels = [
    {"type": "stat", "title": "Firing alerts", "datasource": DS, "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
     "targets": [{"refId": "A", "expr": 'count(ALERTS{alertstate="firing"}) or vector(0)', "datasource": DS}],
     "fieldConfig": {"defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}}, "overrides": []},
     "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}}},
    {"type": "table", "title": "Alerts firing now (rules.yml)", "datasource": DS, "gridPos": {"x": 4, "y": 0, "w": 12, "h": 4},
     "targets": [{"refId": "A", "expr": 'ALERTS{alertstate="firing"}', "instant": True, "format": "table", "datasource": DS}],
     "transformations": [{"id": "organize", "options": {"excludeByName": {"Time": True, "Value": True, "__name__": True, "alertstate": True}}}]},
    {"type": "stat", "title": "Replay clock (UTC)", "datasource": DS, "gridPos": {"x": 16, "y": 0, "w": 4, "h": 4},
     "targets": [{"refId": "A", "expr": "max(solarsentinel_replay_time_seconds) * 1000", "datasource": DS}],
     "fieldConfig": {"defaults": {"unit": "dateTimeAsIso", "color": {"mode": "fixed", "fixedColor": "text"}}, "overrides": []},
     "options": {"colorMode": "value", "reduceOptions": {"calcs": ["lastNotNull"]}}},
    {"type": "stat", "title": "Scorer up", "datasource": DS, "gridPos": {"x": 20, "y": 0, "w": 4, "h": 4},
     "targets": [{"refId": "A", "expr": 'up{job="scorer"}', "datasource": DS}],
     "fieldConfig": {"defaults": {"mappings": [{"type": "value", "options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}}],
                                  "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}}, "overrides": []},
     "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}}},
    ts("S1 data age (h, replay time)", sig("S1_data_age_h"), 0, 4, [mon["s1_data_age_h"]["trigger"]], "h",
       desc="Newest processed slot vs the replay clock. Trigger > 12 h: the dashboard shows STALE."),
    ts("S2 rejected share of batch", sig("S2_reject_share"), 8, 4, [mon["s2_input"]["max_reject_share"]], "percentunit"),
    ts("S2 |Br| vs training (30 d)", sig("S2_br_ratio_30d"), 16, 4, mon["s2_input"]["br_ratio_range"],
       desc="Rolling median of each cutout's median non-zero |Br| over the train reference; band from 2015 val."),
    ts("S3 alert share (30 d)", sig("S3_alert_share_30d"), 0, 11, [s3["val_share"] / s3["trigger_factor"], s3["val_share"] * s3["trigger_factor"]], "percentunit"),
    ts("S4 mean p / observed (30 d, revealed)", sig("S4_calibration_ratio_30d"), 8, 11, mon["s4_calibration"]["ratio_range"]),
    ts("S2 quality-flagged share (context)", sig("S2_quality_nonzero_share"), 16, 11, (), "percentunit"),
    ts("S5 scorer error share (30 d, driver)", sig("S5_error_share_30d"), 0, 18, [mon["s5_scorer"]["max_5xx_share"]], "percentunit"),
    ts("S5 p95 latency (ms, driver round trip)", sig("S5_p95_ms_30d"), 8, 18, [mon["s5_scorer"]["p95_ms"]], "ms"),
    ts("Scorer p95 latency inside the container (ms)",
       'histogram_quantile(0.95, sum by (le) (rate(scorer_request_seconds_bucket{endpoint="/v1/score"}[1m]))) * 1000', 16, 18,
       [mon["s5_scorer"]["p95_ms"]], "ms", legend="p95"),
    ts("Scorer requests by status", 'sum by (code) (rate(scorer_requests_total{endpoint=~"/v1/.*"}[1m]))', 0, 25, (), "reqps", "{{code}}", w=12),
    ts("Rejected inputs by reason", "sum by (reason) (rate(scorer_rejects_total[1m]))", 12, 25, (), "reqps", "{{reason}}", w=12),
]
dash = {"uid": "solarsentinel", "title": "SolarSentinel monitoring", "tags": ["solarsentinel"], "timezone": "utc",
        "schemaVersion": 39, "version": 1, "editable": False, "refresh": "5s", "time": {"from": "now-30m", "to": "now"},
        "description": "Plan section 8: signals S1-S5 (replay time, from the replay driver) and the scorer's own metrics. "
                       "Red lines/areas are the triggers in configs/experiments.yaml m5.monitoring.",
        "panels": [dict(p, id=i + 1) for i, p in enumerate(panels)]}
out = ROOT / "deploy/grafana/dashboards/solarsentinel.json"
out.write_text(json.dumps(dash, indent=1) + "\n")
print(out, len(panels), "panels")
