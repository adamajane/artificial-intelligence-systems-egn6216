#!/bin/sh
# Replay service (C7b + C9): fetch the data set over public HTTPS (sha256-checked), then replay it through the scorer.
# Every container start is a new run in the store (run ID = split + UTC start time unless REPLAY_RUN_ID is set).
set -eu
python -m solarsentinel.serve.fetch ${FETCH_SETS:-dev}
SPLIT="${REPLAY_SPLIT:-val}"
RUN_ID="${REPLAY_RUN_ID:-compose_${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}"
set -- --split "$SPLIT" --run-id "$RUN_ID" --kind "${REPLAY_KIND:-demo}" --reason "${REPLAY_REASON:-docker compose replay}" \
  --scorer-url "${SCORER_URL:-http://scorer:8000}" --slot-seconds "${SLOT_SECONDS:-0}" --metrics-port 9101 --hold "$@"
if [ -n "${REPLAY_OUTCOMES:-}" ]; then set -- "$@" --outcomes "$REPLAY_OUTCOMES"; fi
# shellcheck disable=SC2086
exec python -m solarsentinel.serve.driver "$@" ${REPLAY_EXTRA:-}
