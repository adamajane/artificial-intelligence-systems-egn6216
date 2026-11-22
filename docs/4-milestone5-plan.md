# SolarSentinel: Milestones 5 and 6 plan (brief for Claude Code)

_Milestones 5 and 6, "Final System: Deployment, Testing & Operational Readiness", submitted together as one Word
document. Turn the selected AI component into a system that runs outside the notebook, test it as deployed, run the
final test once, implement monitoring, and give Adam the evidence to close every risk and trust record honestly._

**Deadline mode.** Adam submits on **Monday 2026-11-23 at 23:59 (local time, EST)**. As in Milestone 4, Claude Code
does the code, runs and evidence; Adam gives the work report to a separate assistant, who writes the document. So
**the work report is the main deliverable**. If a number, configuration, commit, screenshot or log is not in the
report, it cannot go in the document. There are two reports: an interim one on **Saturday at 23:00** and the main
one on **Sunday at 22:00** (section 14).

**How to work:** follow `CLAUDE.md`. In short: plan first and get approval; ask before any JSOC export, commit,
push, bucket upload or GCP deployment; numbers come from code; replay (2016–17) is scored **once**, after the
pre-registration in section 5 is committed. Read first: `CLAUDE.md`, `docs/3-milestone4-plan.md`,
`notes/work-reports/2026-11-02_milestone4.md`, `notes/work-reports/2026-11-20_project-state.md`,
`configs/data.yaml`, `configs/experiments.yaml`, `src/solarsentinel/models/`, `src/solarsentinel/data/cutouts_v2.py`.
Update the "current step" pointer in `CLAUDE.md` to this file and fix its stale `docs/data-pipeline-plan.md` link
(ask first).

**Requirement IDs** used below are the ones in the project document: FR1–FR4, PR1–PR5, HR1–HR4 (2.2), T1–T5 (3.3),
R1–R9 (5.3), D1–D5 (2.7), G1–G4 and F1–F5 (4.2). Their full text is quoted where it matters.

---

## 0. Timeline (local time, EST) and priorities

| When                 | Work                                                                                                                                                                                                          | Hard stop                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Fri night**        | Replay target list + dry run → **Adam approves → export starts in the background** (4.1). While it runs: retrain and freeze the bundle (2), CPU parity check (2.2), upload `processed/v2/` (approval) | Export running before Adam goes to bed       |
| **Sat morning**      | Convert replay FITS to cutouts, inputs-only checks (4.2–4.4). Clip-constant control and operator view (3). Write the pre-registration (5) → **Adam approves → commit**                                        | Pre-registration committed by 13:00          |
| **Sat afternoon**    | Scorer API, prediction store, replay driver, tracker (6.2–6.4), with pytest. Dashboard on the 2015 dev stream (6.7)                                                                                           | —                                            |
| **Sat evening**      | Compose stack (6.6). **Final test, once** (7). Dashboard switches to the replay store. **Interim work report**                                                                                               | Report at 23:00                              |
| **Sun morning**      | Monitoring: Prometheus + Grafana, injected failures (8). Security controls (9). Walkthrough build ready for Adam (12)                                                                                          | Walkthrough build ready by 13:00             |
| **Sun afternoon**    | Deployed-system tests and latency (10). Cloud Run (11). **Adam runs the 3 walkthroughs** (12). One small fix from the findings, if there is one                                                                | Stop launching new work at 20:30             |
| **Sun evening**      | Commit and push (approval). **Main work report**                                                                                                                                                               | Report at 22:00                              |
| **Mon until 12:00**  | Buffer only: fixes the document writer asks for, a report addendum. No new features                                                                                                                            | Code freeze 12:00                            |
| **Mon 12:00 → 23:59** | Adam and the writing assistant write the document                                                                                                                                                             | Submitted by 23:59, aim for 21:00            |

**Must:** frozen bundle with hash; replay cutouts; committed pre-registration; final test (PR1, PR2, R9, slices);
scorer API + prediction store + replay driver + dashboard with the failure states and both display modes; compose
stack that runs from a clean clone; the five 8.5 tests plus measured latency; one demonstrated security control; at
least one monitoring signal recorded in SQLite **and** Prometheus with one injected failure; walkthrough build and
scenario switches; the two work reports.
**Should:** Grafana dashboard and alert rules; all five signals; Cloud Run; PR3 timing (Playwright) and PR4
clean-environment timing; clip-constant control; FITS input endpoint; "Flag as questionable" button.
**Could (only if every Must and Should is done):** prequential recalibration analysis (5.6); stale links and README
provenance rows; uploading the 3,850 raw FITS.

**Fallbacks:** if the replay export fails or stalls, run the system and the test on the replay cutouts that finished
and report the coverage, by month and by positives. If Grafana fights back, the SQLite metrics table plus the
Prometheus alert page is enough (the brief asks for a mechanism that records something, not a platform). If Cloud
Run fails, the local stack is the deliverable and 8.3 says so. Drop Should items before cutting any Must.

## 1. Decisions (resolved with Adam, 2026-11-20)

| Topic                    | Decision                                                                                                                                                         | Reason                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model to deploy          | **Retrain ResNet-18, padded input, seed 42, the exact M4 recipe**, and save it this time                                                                          | The M4 weights were never saved. Retraining on train + val would leave no clean data for calibration and the threshold                                                                                                                            |
| Training data            | **Same 3,117-image training set as M4.** No full-dataset training before the deadline                                                                             | The full download is projected at ~25 h at 1,202 records/h with one pending JSOC export, so the bottleneck is the export queue, not compute. A new training set would also reopen the selection that feedback point 1 asks to lock. Goes in 9.7 as the top next step |
| Training compute         | **Laptop (`mps`)**. Vertex AI was evaluated and is not used                                                                                                       | One retrain is about 20 epochs × 5.5 s ≈ 2 min. Update D2: Vertex stays the option if full-set training happens (≈ 44 s per epoch extrapolated, unverified)                                                                                       |
| Deployment calibration   | **One Platt map fitted on all 626 val cases** (prevalence-weighted, as in E3). The cross-fitted M4 numbers stay the reported development results                  | Deployment needs one map; the threshold is already chosen on val                                                                                                                                                                                   |
| Replay scope             | **All 5,101 v2 replay observations**                                                                                                                              | The dashboard and the test both need them; about 4.2 h at the M4 throughput (projection)                                                                                                                                                         |
| Serving                  | **FastAPI scorer service** (new component), called by a replay driver; results in SQLite; Streamlit dashboard reads SQLite                                        | Gives a real endpoint for the 8.4 controls (access, input validation, size limits), a `/metrics` page for Prometheus, and honest end-to-end latency. Changes D1: scoring is still batch per 6-h slot, but goes through an API                     |
| Dashboard                | **Streamlit**, as designed in 4.2–4.3                                                                                                                             | Unchanged from the plan                                                                                                                                                                                                                           |
| Monitoring               | **Prometheus + Grafana** in the compose stack, plus a SQLite metrics table as the durable record                                                                  | Architecture v1 already named Prometheus/Grafana (C9, Increment 2); SQLite keeps evidence even if a panel breaks                                                                                                                                  |
| Hosting                  | **Local `docker compose` is the system of record.** Cloud Run adds a public, read-only dashboard link, and a scorer reachable only with Google sign-in           | The grader can open a link without installing anything. Update D4 and 2.5                                                                                                                                                                          |
| Interaction evaluation   | **Peer walkthrough with 3 friends** (proxy users) + Adam's own cognitive walkthrough and heuristic evaluation                                                    | The M2 plan in 4.4, scaled to 3 participants. Kit: `docs/4-milestone5-walkthrough-kit.md`                                                                                                                                                          |

## 2. Freeze the model bundle (feedback point 1)

The feedback asks for "one exact bundle containing the padded preprocessing, normalization constants, weights,
calibrator, and 0.048 threshold", with every displayed probability traceable to it.

### 2.1 Retrain

- Train ResNet-18 pad, seed 42, `m4.recipe` (20 epochs, final epoch, no early stopping), on the same train cutouts.
  Recompute `c` with the same rule (99.5th percentile of |Br| over non-zero train pixels). It should equal
  **980.5**; a different value means the training data or code differs, so stop and find out why.
- Save `state_dict`, the exact recipe, the data manifest sha256 and the git commit.

### 2.2 Score val through the deployed path (CPU parity)

The deployed scorer runs on CPU in a Linux container; M4 scored on `mps`. So:

1. Write the inference function the scorer will use (`src/solarsentinel/serve/model.py` or similar): load the
   bundle, clip to ±c, divide by c, run the model on CPU, apply Platt.
2. Score the 626 val cases with it **on CPU**, and also with the M4-style `mps` path. Report the maximum absolute
   difference in raw scores. The calibration map and threshold in 2.3 are fitted on the **CPU** scores, because
   those are the scores users will see.

### 2.3 Calibration map and threshold

- Platt on all 626 val cases, prevalence weights from `calibration.prevalence_weights` (natural val: 208 positives,
  4,977 negatives).
- Threshold: the same rule as M4 (`metrics.best_threshold` on the calibrated val scores). The M4 value was
  **0.048**, but it belongs to the old weights, so it is re-picked here and the new value is reported next to 0.048.
- Re-run the three PR2 checks on val for the new model (cross-fitted, as E3 did), to confirm probability mode still
  holds before the test.

### 2.4 Bundle contents and identity

One file, e.g. `models/solarsentinel-resnet18-pad-<short-hash>.pt`, plus a JSON card next to it:
`state_dict`; architecture key; `c` and the clip rule; the v2 cutout rule hash (`v2.rule_hash()`: pad to square,
NaN fill, 128 × 128, float16); Platt coefficients; threshold; val metrics; training data manifest sha256; git
commit; creation time. **Bundle ID = sha256 of the file.** Every prediction row in the store and every API response
carries it (FR3, T3). Check the size against the < 50 MB target (ResNet-18 alone is 44.76 MB fp32).

### 2.5 Acceptance rule (decided before the retrain)

Accept the seed-42 retrain if its val TSS (CPU path, calibrated, re-picked threshold) lies within **0.66–0.72**
(M4: 0.692 ± 0.014 over 3 seeds, so about ±2 sd). If it falls outside, **stop and ask Adam**. Do not try other seeds
until one passes; that would be selecting on val a second time.

### 2.6 Where it lives

`models/` is gitignored. Upload the bundle and card to `gs://solarsentinel_ai-systems-egn6216/models/` (ask first)
so a clean clone can fetch them over public HTTPS without credentials. Upload `processed/v2/` in the same step
(ask first): `uv run python -c "from solarsentinel import storage; print(storage.upload('processed/v2'))"`.

## 3. Pad vs stretch: what E2 isolates (feedback point 2, and "keep thinking about")

The arms used different clip constants (pad 980.5, stretch 1,049) because `c` was computed per arm.

- **Control run (Should):** retrain the stretch arm with `c` fixed at **980.5**, seeds 42/0/1, and report the paired
  pad − stretch TSS difference with the HARP bootstrap, next to the M4 result (−0.013 [−0.059, +0.042]). Written
  into the pre-registration **before** it runs: this run is for attribution only. **The selection stays padded
  ResNet-18 whatever it shows**, and the result is reported as a qualification of the M4 claim.
- **Operator view (Should, minutes):** for pad and stretch at their own val thresholds, report what an operator
  would live with: alerts per day, the share of alerts followed by a flare, and missed flares per month
  (`m4.task_view`). That puts the PR-AUC gap (0.349 vs 0.466) in operator terms, as the feedback asks.

## 4. Replay inputs

### 4.1 Target list and export (approval needed)

All 5,101 v2 replay observations, built the same way as the M4 list (`targets.py`), into its own targets file. Dry
run first: print the number of records, the monthly chunks and the projected time, and check that no request goes
over `max_recordset_chars`. **Adam approves before the export starts.** The export is resumable; log progress to
`data/logs/`.

### 4.2 Conversion

Extend `cutouts_v2.py` (or add a small `cutouts_replay.py`) so that replay is written to **`processed/v2/replay/`**
with its own shards, `index.parquet` and manifest. **The existing train/val shards and `processed/v2/manifest.json`
must not change**; check that their hashes are identical afterwards. Same pad rule, NaN fill, resize and dtype.

### 4.3 Coverage states for the dashboard

The dashboard must show NOT ASSESSED regions (HR2, T4, R5): NOAA-numbered regions outside ±60°, and slots with no
usable record (1.3% of slots stay missing, 6.4). Check what `processed/v2/observations.parquet` holds. If it only
has assessed rows, derive NOT ASSESSED from the SHARP keyword table already on disk. Report the counts.

### 4.4 Inputs-only checks (no model output)

Rows converted vs targets, NaN fraction and pad fraction distributions, and `quality_nonzero` share (M4: 2,869 of
5,101, 56.2%). Also look up what the QUALITY bits 0x80 and 0x10080 mean in the JSOC/HMI documentation, so R9 can say
what the flag is, not just how often it appears. No model output on replay until section 7.

## 5. Pre-registration of the final test

Write `docs/4-milestone5-preregistration.md`. **Adam approves it, and it is committed (and pushed) before any
replay observation is scored.** The commit hash is the evidence that the rules came first. Contents:

### 5.1 What is frozen

The bundle ID (2.4), the threshold, display-mode rule, data version (`processed/v2` + replay manifest hash), and the
evaluation code commit.

### 5.2 Primary results (the targets from 2.1/2.2)

- **PR1:** test TSS ≥ 0.5 and above persistence, with 95% HARP-bootstrap CIs. Persistence and climatology are scored
  on exactly the same replay rows (this also fills the new 7.2 "Baseline" box). Report distinct flaring regions and
  flare events next to positive observations.
- **PR2:** BSS > 0 against climatology, plus the three checks as written in 2.2 (mean p / observed rate in
  [0.5, 2]; slope CI contains 1; ≥ 4 of 5 equal-count bins consistent). Replay is at natural prevalence, so no
  weights. **If any check fails, the dashboard runs in ranking mode** (4.3). That rule already exists; restate it.
- **R9:** the **clean subset is primary**, with all records and flagged records alongside (already decided in 5.3).
- **R1 check:** val–test TSS gap > 0.15, or beating persistence on val but not on test → audit split and labels and
  report it as a finding. Note that val TSS is optimistic by construction (threshold picked on the same cases).

### 5.3 Where it works and where it fails (slices)

Each slice reports counts (observations, positives, flaring HARPs). Recall, false-alarm share and TSS are shown
**only when the slice has at least 10 positives**; otherwise the counts are given as they are ("3 of 7 caught").
Slices:

| Slice                                         | Why                                                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| USFLUX tercile (same cut points as M4)        | Known weakness: val recall 0.557 in the lowest tercile vs 0.986 above                               |
| Flare size: M1–M2, M2–M5, ≥ M5 (incl. X)      | Near-threshold flares are where misses were on val (`errors.flare_bin`)                             |
| `quality_nonzero` yes/no                      | R9                                                                                                  |
| Distance from central meridian: 0–30°, 30–60° | Projection effects towards the ±60° cut                                                             |
| Multi-NOAA HARP yes/no                        | 24% of observations; label-attribution risk (R2)                                                     |
| Thin regions (aspect ≥ 2)                     | What padding was meant to preserve (E2)                                                             |
| **With and without AR 12673**                 | 40 of the 62 M/X flares in 2017 came from one region (PR1 note), so one region can carry the result |
| False alarms by region size                   | The val pattern: large regions that produce only C flares                                           |

### 5.4 Calibration at the operating prevalence (feedback point 3)

Write down the expectation before the test: the Platt map was fitted at the 2015 base rate (208 / 5,185 = 4.0%),
and replay runs at **74 / 5,101 = 1.45%**. The expected result is **over-forecasting** (mean p / observed rate above
1, possibly above the 2× trigger). Signal: mean p / observed rate, overall and over rolling 30-day windows. Trigger
and response: outside [0.5, 2] → the R6 response (report in replay; recalibrate in live mode) and PR2 check 1 fails
→ ranking mode.

### 5.5 What would make us reconsider (the feedback's "one question")

Answer it in writing, before the test: padded ResNet-18 is called into question if PR1 fails (TSS < 0.5, or the
CI of the difference to persistence includes 0) or the R1 trigger fires. The displayed probabilities are called
into question if PR2 fails. **In both cases, no model or threshold is changed using the test.** The outcome is
reported as a finding, the dashboard follows the mode rule, and the next step (stretch arm, hybrid image + SHARP
model, recalibration on new data) goes into 9.7.

### 5.6 Re-runs and secondary analyses

A re-run of the test is allowed only to fix a pipeline bug that changed inputs or outputs (not a modelling
choice). Every run is logged with a run ID and a reason, and all of them are reported. **Could:** a prequential
recalibration analysis (at each slot, shift the Platt intercept using only outcomes that are already known, i.e.
older than 24 h, over the previous 30 days). Labelled secondary; it never replaces the primary result or the mode
decision.

## 6. The system (8.1–8.3)

### 6.1 Components (Architecture v2, mapped to v1)

| ID     | Component                         | Runs where                                | Talks to                                                      |
| ------ | --------------------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| C1–C4  | Data pipeline (existing)          | Laptop, run on demand                     | JSOC, NOAA → `data/` → public GCS bucket                      |
| C5–C6  | Training + bundle (2)             | Laptop (`mps`)                            | → bundle in GCS `models/`                                     |
| **C7a** | **Scorer API** (FastAPI, new)     | Container, CPU                            | Loads the bundle at start; `/v1/score`, `/healthz`, `/metrics` |
| **C7b** | **Replay driver** (new)           | Container, one-shot or streaming          | Reads replay cutouts → calls C7a → writes C8                  |
| C8     | Prediction store                  | SQLite on a compose volume                | Written by C7b, C9; read by C10                               |
| C9     | Outcome tracker + monitor         | Inside C7b, or its own small process      | Joins outcomes after 24 h of replay time; writes metrics      |
| C10    | Dashboard (Streamlit)             | Container                                 | Reads C8 (read-only)                                          |
| **C11** | **Prometheus + Grafana** (new)    | Containers                                | Scrape C7a and C9; alert rules                                |

Architecture v2 figure: produce a mermaid source **and** a rendered PNG in `docs/figures/m5/`, with runtime
locations, connections, storage, external dependencies and their failure points (JSOC, GCS public HTTPS at build
time, Cloud Run), and monitoring drawn on it. Keep v1 next to it for comparison.

### 6.2 Scorer API contract

- `POST /v1/score`: body = one 128 × 128 v2 cutout (`.npy`, float16 or float32) + metadata as JSON fields
  (`harpnum`, `t_rec`, `quality`, `source`). Returns `p` (calibrated), `alert` (at the bundle threshold), `mode`,
  `bundle_id`, `threshold`, `request_id`, `latency_ms`.
- `POST /v1/score/fits` (**Should**): a raw SHARP CEA Br FITS, preprocessed with the same `cutouts_v2` functions,
  so the C2 code is reused rather than copied.
- `GET /healthz` (bundle loaded, bundle ID), `GET /metrics` (Prometheus).
- Validation (9) happens before anything reaches the model. Errors are JSON with a reason code.

### 6.3 Prediction store (SQLite) schema

`predictions` (one row per region per slot: `harpnum`, `noaa_ars`, `t_slot`, `t_rec`, `p`, `alert`, `mode`,
`bundle_id`, `threshold`, `source`, `quality`, `request_id`, `scored_at`); `coverage` (per slot: assessed,
not-assessed and no-data regions, with reasons); `outcomes` (label, flare class and time, revealed only once replay
time passes `t_slot + 24 h`); `metrics` (signal name, replay time, wall time, value; section 8); `feedback`
(**Should**: "Flag as questionable" notes, max 500 characters, with the prediction key). FR3 test: 0 null provenance
fields.

### 6.4 Replay driver and tracker

Walks the replay slots in time order and sends each slot's batch to the scorer. Speed is configurable (as fast as
possible for the final test; for example 1 slot per 2 s for the monitoring demo). It writes coverage per slot. The
tracker reveals outcomes after 24 h of replay time and computes the Tracking-page figures (running hit rate,
false-alarm share, TSS, Brier) and the monitoring signals (8). Fault switches for tests and the walkthrough (env
vars or CLI flags): `--skip-slots`, `--stall-after`, `--scale-br 2.0`, `--corrupt-every N`.

### 6.5 Dashboard

As designed in 4.2–4.3, built for the primary scenario (replay date 2017-09-05 12:00 UTC, AR 12673 before its X9.3
flare):

- **Every page (HR1):** "Research prototype, not an official forecast" banner with the NOAA SWPC link; mode chip
  (REPLAY); data time and data age; bundle ID.
- **Overview:** sortable region table, the accessible primary view (rank, region, calibrated p with natural-frequency
  wording, ▲ ALERT text + icon); NOT ASSESSED count with reasons; disk map as an extra; date stepper.
- **States (HR2):** NO DATA, STALE ("last known", with age) and NOT ASSESSED must look and read differently from
  "no alerts". An empty alert list is never shown without a state message.
- **Threshold control (HR3):** custom-value chip, one-click reset, and the 2015 operating point next to it (alerts
  per day, share followed by a flare, missed flares per month), plus the one sentence explaining that rarer flares in
  2016–17 raise the false-alarm share even if the model is unchanged.
- **Region detail:** Br cutout, key SHARP parameters, base rate, what an alert means at this threshold, provenance,
  and the small-region notice with its sample count (feedback point 3: recall 0.557 in the lowest-flux third, with
  n).
- **Tracking:** running figures over the replay period, rolling predicted vs observed rate.
- **About / model card:** training period, val and test metrics with CIs, blind spots (small regions, limb, quality
  flags), intended use, bundle ID.
- **Modes:** probability mode and ranking mode both built; which one runs follows the PR2 result (5.2).
- **Accessibility (HR4):** palette and contrast ≥ 4.5:1; status never by colour alone; a greyscale screenshot of each
  page.
- **Walkthrough scenario switch (for 12):** `?scenario=stale` and `?scenario=not_assessed`, or env vars, so Adam can
  inject the two states without restarting.

### 6.6 Packaging

- Two images: `scorer` (Python 3.13 slim, **CPU-only torch** from the PyTorch CPU wheel index, timm, numpy, pyarrow,
  fastapi, uvicorn, prometheus-client, astropy and scikit-image for the FITS path) and `dashboard` (streamlit,
  pandas, pyarrow, plotting; no torch). A separate runtime requirements file, because `uv.lock` pulls the default
  torch build, which brings CUDA libraries on Linux.
- `compose.yaml`: scorer, replay, dashboard, prometheus, grafana; a named volume for SQLite; **all host ports bound
  to 127.0.0.1**.
- Data and bundle are fetched at build or first start over **public HTTPS** from the bucket, with sha256 checks
  against the manifests. No credentials anywhere in the runtime.
- `.dockerignore` excludes `.env`, `data/`, `notes/`, `experiments/`, `.git`.
- Apple Silicon builds arm64 images; Cloud Run needs linux/amd64 (11).
- README "Run it" section: from `git clone` to the dashboard in a few commands; tested from a clean clone (10).

### 6.7 Building the dashboard without looking at the test

Until section 7 runs, develop the dashboard against a **2015 dev stream**: the frozen bundle's predictions on the
626 val cases, loaded into the same store format. Replay predictions appear in the dashboard only after the final
test has run.

## 7. Final test, once (9.1)

After the pre-registration is committed: run the replay driver over all replay observations through the deployed
scorer (compose stack, CPU), with run ID `m5_final_test_<bundle-short-hash>`. The evaluation script reads the
store, not a notebook variable, so the test result is a test of the deployed system. Outputs to
`experiments/results/m5/`: metrics JSON (5.2), slices (5.3), calibration (5.4), reliability figure, and an error
gallery of the largest misses and confident false alarms (as in M4). Run the same evaluation code on the val dev
stream first, to show it reproduces the 2.3 numbers.

## 8. Monitoring (9.3–9.4)

Five signals at most, all computed in replay time by C9, written to the SQLite `metrics` table and exported as
Prometheus gauges or counters. Alert rules in Prometheus (`rules.yml`), panels in a provisioned Grafana dashboard
(JSON in the repo; anonymous read-only viewer, admin password from an untracked env file).

| Signal                                           | Expected range (fill from val or design)            | Trigger                                                  | Response                                                      | Links       | Priority |
| ------------------------------------------------ | --------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- | ----------- | -------- |
| S1 Data age (newest scored slot vs replay clock) | ≤ 6 h                                               | > 12 h                                                   | STALE banner, "last known" values; check ingest               | PR5, R5, T4 | Must     |
| S2 Input integrity: rejected inputs per batch, `quality_nonzero` share, median non-zero \|Br\| vs train reference | rejects ≈ 0; \|Br\| ratio near 1 | rejects > 10% of a batch, or \|Br\| ratio outside [0.8, 1.25] | Hold alerts with an "input check" notice; investigate         | R7, R9, T4  | Should   |
| S3 Alert share per assessed region-slot, 30 days | Around the val share at the frozen threshold        | > 2× or < 0.5× the val share                             | Review; read together with S4                                 | R6, T2      | Should   |
| S4 Mean p / observed rate, rolling 30 days       | [0.5, 2]                                            | Outside [0.5, 2]                                         | Report (replay) / recalibrate (live); ranking mode if PR2 fails | R6, PR2, T2 | Must     |
| S5 Scorer error share and p95 latency            | errors ≈ 0; p95 well under 100 ms per region        | 5xx > 1%, or p95 > 100 ms                                | Inspect logs; restart; the dashboard shows the failure state  | D1, PR3     | Should   |

**Injected failures (the 9.4 evidence),** each with a SQLite excerpt, a Grafana screenshot showing the signal
before and after, the Prometheus alert firing, and what the dashboard showed:

1. **Stalled feed** (`--stall-after`): S1 fires, and the dashboard shows STALE within one slot (T4 evidence).
2. **Unit or calibration change in the input** (`--scale-br 2.0` from some date on): S2 fires; report what S3 and
   S4 do.
3. S4 on the real replay stream is **not injected**: it shows whatever the base-rate shift actually does, and that
   is the honest "observed change" for 9.2.

Note the PR5 boundary: live mode (Increment 1) was not built, so S1 is demonstrated on replay time, not over two
weeks of wall time.

## 9. Security controls (8.4)

| Area    | Local stack                                                                                                                                    | Cloud Run                                                                                     |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Access  | Ports on 127.0.0.1 only (T5); the scorer is reachable inside the compose network, plus 127.0.0.1 for tests                                     | Dashboard public, read-only. **Scorer: `--no-allow-unauthenticated`** (Google sign-in required), `--max-instances 1` |
| Secrets | The runtime needs none. `JSOC_EMAIL` stays in `.env`, excluded by `.dockerignore`. Gitleaks scan of the repo **and** of the built image filesystem | None in the image; no keys                                                                    |
| Inputs  | Body size limit checked before parsing → 413; dtype, shape (128 × 128), finite values, NaN share, metadata schema (Pydantic) → 422             | Same code, plus Cloud Run's own request limits                                               |
| Logging | Structured JSON per request: request ID, bundle ID, region, slot, outcome or reject reason, latency. No secrets. The system holds no personal data; feedback notes are length-limited, and the UI asks users not to type personal data | Cloud Logging receives the same lines                                                         |

**Demonstrated control (Must):** input validation and size limits, as a short transcript (curl or pytest output):
oversized body → 413, wrong shape → 422, NaN-only array → 422, valid → 200 with the bundle ID. **Should:** on Cloud
Run, the scorer without a token → 403 and with `gcloud auth print-identity-token` → 200. Write down which threat
each control addresses and what production would add (Secret Manager, auth on the dashboard, rate limits, pinned
image digests).

## 10. Tests in the deployed system (8.5)

Automated where possible (`tests/test_api.py`, `tests/test_system.py` against the running stack), with every result
saved to `experiments/results/m5/tests/`.

| Test                   | What runs                                                                                                                                         | Checks                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Valid case             | A val cutout through the API: matches the 2.2 CPU-path score (abs diff < 1e-5) and carries the bundle ID                                          | FR3, T3                                   |
| Edge case              | A thin region (aspect ≥ 4), a cutout with high NaN share, a quality-flagged record, and a region outside ±60° (→ NOT ASSESSED, never scored)       | HR2, T4, R9                               |
| Malformed input        | Truncated FITS (Should) or `.npy`, wrong shape, NaN-only array, wrong content type, oversized body, bad metadata                                  | 8.4, T4                                   |
| Integration            | Driver → API → SQLite → dashboard for 2017-09-05 and two more fixed dates; 0 null provenance fields; Tracking figures recomputed from C8 match the dashboard to 3 decimals | FR3, FR4                                  |
| **Deliberate failure** | Stop the scorer in the middle of the replay (`docker compose stop scorer`), then withhold one batch. Screenshot **what the user actually saw**, then recover | HR2, R5, T4                               |

**Performance (goes into 9.1):**

- End-to-end latency per slot: from the driver reading a slot's cutouts to its rows being committed in SQLite. p50
  and p95 over the whole replay, and per region (target from 2.3/2.6: < 100 ms per region; M4 model-only figure: 5.74
  ms).
- API request latency inside the container (p50/p95), and on Cloud Run with and without a cold start.
- PR3: a Playwright script, 100 view changes (next date, open a region), p95 ≤ 2 s.
- PR4: fresh clone in a new folder, Docker images and cache removed, time from `git clone` to the first dashboard
  page (target ≤ 10 min, no GPU, no credentials). A second machine would be stronger; say which was used.
- Throughput: slots per minute and regions per second over the replay.

**Bugs and integration problems:** keep a running log (`experiments/results/m5/integration-log.md`): what broke when
components met, what was changed, and the commit. This is the raw material for 8.6, so record the small things too
(dtype mismatches, path assumptions, arm64 vs amd64, cold starts, timezone handling).

## 11. Cloud Run (Should)

Ask before enabling APIs or deploying. Region `us-east1`, next to the bucket.

- Build linux/amd64 images with Cloud Build (`gcloud builds submit`) or `docker buildx --platform linux/amd64`, and
  push to Artifact Registry.
- **Dashboard service:** public, read-only, with the SQLite store from the final test baked into the image (Cloud
  Run disk is not persistent, so the "Flag" button is disabled there, with a note). Streamlit uses websockets, so
  check that the page works through the Cloud Run URL.
- **Scorer service:** IAM-only (9), `--max-instances 1`, 1 vCPU, enough memory for torch (try 1 GiB).
- Cost: request-based billing has a monthly free allowance (2 million requests, 180,000 vCPU-seconds, 360,000
  GiB-seconds per Google's pricing page, checked 2026-11-20), and services scale to zero when idle. Artifact Registry
  storage for ~1–2 GB of images costs a little; measure it. The budget alert with the $20 cap stays. Record measured
  GCP cost for November (Adam reads it from the billing console) for 9.1.
- If it fails or eats more than ~2 h, stop. 8.3 says the local stack is the system, and Cloud Run was attempted.

## 12. Interaction evaluation support (4.4)

Adam runs the walkthroughs, so the observations are his. Claude Code's part:

- A walkthrough build by **Sunday 13:00**: the replay store loaded, the date set to 2017-09-05 12:00 UTC, and the two
  scenario switches working (6.5).
- Adjust `docs/4-milestone5-walkthrough-kit.md` to the built UI: real page names, where the controls are, how to
  switch scenarios. Do not change the tasks or success criteria; they were planned in M2.
- After the sessions, Adam gives Claude Code his completed sheets. Implement **at most one** small change if a finding
  calls for it (a wording or state-design fix), re-check it, and log it. Larger findings are recorded as unresolved.
- R5's trigger: if any participant reads NO DATA or NOT ASSESSED as safe (G3), the redesign of those states becomes
  the one change.

## 13. Housekeeping (Could, unless cheap)

Fix the 9 stale `docs/data-pipeline-plan.md` links; README provenance rows for the M4 cutouts, `processed/v2/`,
replay and the bundle; ask about the 6 `_dryrun` files and the empty `v1/` folder at the repo root.

## 14. Work reports: what the document needs

Same format as `CLAUDE.md`. **Interim report Saturday at 23:00** (bundle, pre-registration, final test, system
state). **Main report Sunday at 22:00.** Beyond the usual sections, the main report must give the writer:

1. **Revision table (first page).** One row per feedback point, with what changed and where. Note the renumbering:
   the final template adds **7.2 Baseline**, so M4's 7.2–7.7 are now 7.3–7.8. Points: (1) bundle lock → 7.4, 7.6;
   (2) clip constants → 7.3; (3) calibration and low-flux checks → 7.5, 7.8; keep-thinking (PR-AUC in operator terms)
   → 7.6; the one question → 7.8 and the pre-registration.
2. **Section 8:** environment facts (versions, image sizes, where each part runs); Architecture v2 PNG + mermaid; the
   run instructions as tested; the security transcript; the five-row test table with what actually happened and
   screenshots; latency numbers; the integration log.
3. **Section 9:** a 9.1 table draft with one row per target (PR1, PR2, PR3, PR4, PR5 marked not built, size < 50 MB,
   < 100 ms per region, throughput, cost: estimated vs measured); the signals table as implemented; monitoring
   evidence (SQLite excerpt, screenshots, alert firing); observed changes vs anticipated ones.
4. **Closing evidence for every row of 3.3 (T1–T5) and 5.3 (R1–R9):** what was implemented, what changed, what was
   never implemented and why, with the evidence file for each. Mark which ratings look like they stay High or
   Critical.
5. **2.x updates:** D1 (API), D2 (Vertex evaluated, not used), D4 (Cloud Run), 2.5 cost (measured), 2.3–2.4
   (runtime versions).
6. **Bundle identity:** bundle ID, threshold, Platt coefficients, val metrics of the retrain next to M4's.
7. **Repository link, final commit hash, Cloud Run URL** (if deployed).
