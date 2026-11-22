#!/bin/sh
# Plan 9 demonstrated control: input validation and size limits on the running scorer, as a curl transcript.
# Usage: deploy/tools/security_demo.sh [base URL, default http://127.0.0.1:8000] > transcript.txt
set -u
URL="${1:-http://127.0.0.1:8000}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
uv run python - "$T" <<'PY'
import io, sys, numpy as np, pandas as pd
from pathlib import Path
from solarsentinel import storage
t = Path(sys.argv[1])
idx = pd.read_parquet(storage.local_path("processed/v2/cutouts/index.parquet"))
r = idx.iloc[100]
x = np.asarray(np.load(storage.local_path(f"processed/v2/cutouts/{r.shard}.npy"), mmap_mode="r")[r["index"]])
def save(name, a, pickle=False):
    b = io.BytesIO(); np.save(b, a, allow_pickle=pickle); (t / name).write_bytes(b.getvalue())
save("valid.npy", x)
save("wrong_shape.npy", x[:64, :64])
save("nan_only.npy", np.full((128, 128), np.nan, np.float32))
save("oversized.npy", np.zeros((400, 400), np.float32))
save("pickled.npy", np.array([{"payload": "object"}], dtype=object), pickle=True)
(t / "meta.json").write_text('{"harpnum": %d, "t_rec": "%s", "quality": 0, "source": "test"}' % (r.HARPNUM, r.T_REC))
PY
META=$(cat "$T/meta.json")
show() { # name, curl args...
  name="$1"; shift
  echo "### $name"
  echo "\$ curl $*" | sed "s|$T/||g"
  curl -s -o "$T/out" -w "HTTP %{http_code}\n" "$@"
  cat "$T/out"; echo; echo
}
echo "# SolarSentinel scorer: input controls ($(date -u +%Y-%m-%dT%H:%M:%SZ), $URL)"
echo
show "health" "$URL/healthz"
show "valid cutout -> 200 with bundle ID" -F "cutout=@$T/valid.npy" --form-string "meta=$META" "$URL/v1/score"
show "oversized body (640 KB > 256 KB limit) -> 413" -F "cutout=@$T/oversized.npy" --form-string "meta=$META" "$URL/v1/score"
show "wrong shape (64x64) -> 422" -F "cutout=@$T/wrong_shape.npy" --form-string "meta=$META" "$URL/v1/score"
show "NaN-only array -> 422" -F "cutout=@$T/nan_only.npy" --form-string "meta=$META" "$URL/v1/score"
show "pickled object in .npy (never unpickled) -> 422" -F "cutout=@$T/pickled.npy" --form-string "meta=$META" "$URL/v1/score"
show "bad metadata (unknown field, bad t_rec) -> 422" -F "cutout=@$T/valid.npy" --form-string 'meta={"harpnum": 1, "t_rec": "yesterday", "quality": 0, "source": "test", "admin": true}' "$URL/v1/score"
show "wrong content type (JSON) -> 415" -H "content-type: application/json" -d '{"cutout": [1, 2, 3]}' "$URL/v1/score"
show "scorer metrics after the demo (rejects by reason)" "$URL/metrics"
