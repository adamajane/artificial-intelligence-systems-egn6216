# SolarSentinel: pre-registration of the final test (Milestones 5–6)

_Written 2026-11-20/21, before any replay (2016–17) observation was scored and before any replay label was read.
The commit that adds this file is the evidence that the rules came first. Plan: `docs/4-milestone5-plan.md`
section 5. Every rule below is also in `configs/experiments.yaml` (`m5.final_test`), and the evaluation code that
applies them, `src/solarsentinel/models/final_test.py`, is in the same commit._

Items marked **(default)** are Claude Code's reading of a point the plan leaves open. Adam confirmed or changed
them before this file was committed (see the commit message).

## 1. What is frozen

| Item | Value | Source |
| --- | --- | --- |
| Model bundle | `solarsentinel-resnet18-pad-0101774eb973.pt`, bundle ID (sha256) `0101774eb973e9cfa7f297796fe4927f0d8f723058c6a6a6fa74905aa81fe452`, 44.76 MB | `experiments/results/m5/m5_bundle_resnet18_pad_s42_report.json`; `gs://solarsentinel_ai-systems-egn6216/models/` |
| What the bundle holds | ResNet-18 (`resnet18.a1_in1k`, 1 input channel, 1 logit), `c` = 980.5 G, v2 cutout rule (pad to square, NaN → 0, 128 × 128, float16), Platt map a = 0.70597, b = −2.35547, threshold | bundle card (`.json` next to the file) |
| Alert threshold | **0.05262** on the calibrated scale (max-TSS rule on calibrated val scores; M4's 0.048 belonged to the cross-fitted maps) | same report |
| Model identity | The retrain is bit-identical to M4's `m4_E2_resnet18_pad_s42` (max \|Δp\| = 0.0 over the 626 val cases), so the bundle *is* the selected M4 model | `m5_bundle_resnet18_pad_s42_val_scores.parquet` vs `m4_E2_resnet18_pad_s42_val_preds.parquet` |
| Bundle code commit | `2c80012` (not dirty) | bundle card `git_commit` |
| Training data | `processed/v2` manifest sha256 `10816699b251c80f94479a133e0d83f0d8b66e0e48da64459d9b07924ddf580a` (unchanged since M4) | bundle card `data_manifest_sha256` |
| Test inputs | Replay target list `processed/v2/replay/targets.parquet`, sha256 `657e7869ddd8b4234c423c23ee690c39e03411eda058f924643d5bcb0f993d76`: all 5,101 v2 replay observations (183 HARPs, 2016-01-28 12:00 → 2017-12-29 00:00 UTC). All 5,101 FITS exported (2026-11-21 02:12–04:33) and cut out with the v2 rule by the same functions as train/val (`data.cutout_rules`): `processed/v2/replay/manifest.json` sha256 `ffdf4352acd7f0e207233f3b62e11221c9beaed749184bdb6c907c0721cb73dc` (27 files). Inputs-only checks (no label, no model output): 5,101 converted, 2,869 (56.2%) `quality_nonzero`, 8 cutouts with NaN pixels, median pad fraction 0.509, median non-zero \|Br\| 1.055× the train reference (`processed/v2/replay/inputs_check.json`) | `targets --replay`, `cutouts_replay`, plan 4.1–4.4 |
| Display mode rule | Probability mode if PR2 holds on the test (section 3); otherwise ranking mode | plan 5.2, M4 E3 |
| Evaluation code | `src/solarsentinel/models/final_test.py` in this commit. It reads the prediction store (SQLite) written by the deployed system, not a notebook variable | plan 7 |

Nothing in this table is changed using the test. The test runs once (section 7).

## 2. The test set

- **Scored rows:** every v2 replay observation that the deployed system scored: replay driver → scorer API (CPU,
  container) → SQLite store, run ID `m5_final_test_0101774eb973`.
- **Incomplete export (fallback, plan 0):** if some replay cutouts are missing when the test runs, the test runs on
  the rows that have a cutout, and the missing rows are reported by month and by label (counts only). No row is
  dropped for any other reason.
- **Natural prevalence:** every replay observation is downloaded, so no weights are used (val needed them).
- **Labels** come from `processed/v2/observations.parquet` through one gate function
  (`final_test.replay_outcomes`) that refuses to run unless this file is committed and unchanged, and logs every
  call. Persistence uses the same attributed flare list as the labels (`baseline.persistence`).

## 3. Primary results

**Subsets (R9).** The **clean subset is primary**; all records and the quality-flagged records are reported
alongside.
- Clean = scored rows minus `quality_nonzero` rows and minus flagged negatives (`has_unattributed_mx` and label 0).
  This is the M4 rule `v2.label_rule.eval_clean_exclude` (`dataset.clean_mask`). **(default: one clean definition
  for val and test)**
- What the flag means: `experiments/results/m5/quality_bits.md` (bit 0x80 `QUAL_TEMPERROR`, set by JSOC in
  Sept 2017 on records of the first year or so of the "mod L" calibration waiting for reprocessing; 0x10000
  `QUAL_LOWINTERPNUM`).

**PR1 (skill).** On the primary subset, at the frozen threshold 0.05262:
- TSS ≥ **0.5** (point estimate), with a 95% HARP-bootstrap CI (1,000 reps, seed 0);
- **and above persistence:** the paired HARP-bootstrap 95% CI of TSS(model) − TSS(persistence) on the same rows
  lies above 0. **(default: "above" means the CI excludes 0, as plan 5.5 words the failure)**
- Persistence (alert if the same HARP had an attributed ≥M1.0 flare in the previous 24 h) and climatology are scored
  on exactly the same rows. Climatology is a constant forecast, so its TSS is 0; it is the reference for BSS.
- Reported next to the positive observations: distinct flaring regions (HARPs) and distinct flare events.
- PR1 passes only if both conditions hold on the primary subset. All-records and flagged-only results are reported
  with the same metrics and do not change the verdict.

**PR2 (probabilities).** On the primary subset, at natural prevalence (no weights):
- BSS > 0 against climatology. **(default: climatology = the observed rate of the same rows, the M4 definition;
  the fixed 2015 rate 0.0401 is reported as a second baseline row)**
- Check 1: mean p / observed rate in [0.5, 2].
- Check 2: the HARP-bootstrap 95% CI of the calibration slope contains 1.
- Check 3: 5 equal-count bins; the HARP-bootstrap 95% CI of each bin's observed frequency contains its mean p, with
  at most 1 miss.
- **If BSS ≤ 0 or any check fails, the dashboard runs in ranking mode** (rank and alert flag, no probability
  values). This rule already exists (M4 E3); it is restated here. **(default: the mode follows the primary
  subset)**

**R1 (generalization gap).** Val TSS of the bundle is 0.693 [0.573, 0.784] (626 cases, threshold picked on those
same cases, so optimistic by construction). If val TSS − test TSS > **0.15**, or the model beats persistence on val
but not on test, the split and the labels are audited, and the audit is reported as a finding.

## 4. Where it works and where it fails (slices)

Every slice reports observations, positives and flaring HARPs. Recall, false-alarm share (share of alerts not
followed by a flare) and TSS are shown **only when the slice has at least 10 positives**; otherwise the counts are
given as they are ("3 of 7 caught"). Slices use the primary subset and are repeated for all records.

| Slice | Definition (frozen) |
| --- | --- |
| USFLUX tercile | low ≤ 2.342232e22 Mx < mid ≤ 3.853554e22 Mx < high (M4 terciles of the 208 val positives) |
| Flare size (positives) | M1.0–1.9, M2.0–4.9, ≥ M5 incl. X (science-scale peak flux, `errors.FLARE_BINS`) |
| `quality_nonzero` | yes / no (this slice uses all records, since the clean subset has none) |
| Distance from central meridian | \|LON_FWT\| 0–30°, 30–60° |
| Multi-NOAA HARP | `multi_noaa` yes / no |
| Thin regions | long side / short side ≥ 2 (cutout index `orig_nx`, `orig_ny`) |
| AR 12673 | with and without HARP 7115 (AR 12673) |
| False alarms by region size | false-alarm share among negatives per USFLUX tercile (same cut points) |

Also reported, secondary: the operator view at the frozen threshold (alerts per day, share of alerts followed by a
flare, missed flares per month; `m4.task_view` rules), the reliability figure, and an error gallery of the 8 largest
misses and the 8 most confident false alarms.

## 5. Calibration at the operating prevalence (feedback point 3)

**Expectation, written before the test.** The Platt map was fitted at the 2015 base rate (208 / 5,185 = 4.0%).
Replay runs at **74 / 5,101 = 1.45%** (split audit). The expected result is **over-forecasting**: mean p / observed
rate above 1, possibly above the 2× trigger. If so, PR2 check 1 fails and the dashboard runs in ranking mode.

**Signal:** mean p / observed rate, overall and over rolling 30-day windows of replay time (window counts shown;
windows with no positive report the mean p only). **Trigger and response:** outside [0.5, 2] → the R6 response
(report it in replay; recalibrate in live mode), and PR2 check 1 failing → ranking mode.

## 6. What would make us reconsider

- **Padded ResNet-18 is called into question** if PR1 fails (TSS < 0.5, or the paired CI against persistence
  includes 0), or if the R1 trigger fires.
- **The displayed probabilities are called into question** if PR2 fails.
- **In both cases no model, threshold or calibration is changed using the test.** The outcome is reported as a
  finding, the dashboard follows the mode rule, and the next step (stretch arm, hybrid image + SHARP model,
  recalibration on new data) goes into 9.7.

## 7. Re-runs and secondary analyses

- The test runs **once**. A re-run is allowed only to fix a pipeline bug that changed inputs or outputs (for example
  a wrong cutout, a dropped slot, a store write error), never a modelling choice. Every run gets a run ID and a
  reason in the run log, and all runs are reported.
- The same evaluation code runs first on the 2015 val dev stream and must reproduce the bundle's val numbers (TSS
  0.693 at 0.05262 on 626 cases, with val prevalence weights) before it touches replay.
- **Clip-constant control (plan 3, attribution only).** Before the test, the stretch arm is retrained with `c` fixed
  at 980.5 G (seeds 42/0/1) and compared with the M4 pad runs (paired HARP bootstrap on val). It qualifies the M4
  pad-vs-stretch claim. **The selection stays padded ResNet-18 whatever it shows.** It uses train and val only.
- **Could:** a prequential recalibration analysis (at each slot, shift the Platt intercept using only outcomes older
  than 24 h from the previous 30 days). Labelled secondary; it never replaces the primary result or the mode
  decision.
