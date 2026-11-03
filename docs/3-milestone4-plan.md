# SolarSentinel: Milestone 4 plan (brief for Claude Code)

_Milestone 4, "AI Component & Experimentation". Run the experiments that decide which AI component the system
uses, show the evidence (including where it fails), and close the three Milestone 3 feedback items._

**Deadline mode.** Adam submits the Word document tonight at **23:59**. Claude Code does the code, runs and
evidence; Adam then gives the work report to a separate assistant, who writes the document. So **the work report
(section 9) is the main deliverable of this session**: if a number, configuration, commit or figure is not in the
report, it cannot go in the milestone. It is due by **22:00**, whatever state the experiments are in.

**How to work:** follow `CLAUDE.md`. In short: plan first and get approval; ask before any JSOC export, commit,
push or bucket upload; numbers come from code; replay (2016–17) stays untouched. Read first: `CLAUDE.md`,
`docs/2-model-experiment-plan.md` (first exploration), `notes/work-reports/2026-10-26_first-model-exploration.md`,
`configs/data.yaml`, `configs/experiments.yaml`, `src/solarsentinel/models/`, `src/solarsentinel/data/cutouts.py`.
Update the "current step" pointer in `CLAUDE.md` to this file (ask first).

---

## 0. Timeline (local time) and priorities

| Time          | Work                                                                                                                                                                                                                                                                                                                           | Hard stop                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| 18:15 → 18:35 | New target list (2.4), dry-run, **Adam approves → export starts in the background**, `--stop-at 20:45`                                                                                                                                                                                                                         | Export running by 18:40      |
| 18:35 → 20:45 | While it downloads: v2 split + audit (2.1), frozen label rule (2.2), padded conversion + inspection + manifest on the 1,432 FITS already on disk (2.3), persistence baseline (3.1), code for E1–E3, calibration and error analysis. **Dry-run everything on the 1,260 existing images** (that run is also the fallback result) | —                            |
| 20:45 → 21:00 | Build padded (v2) and stretched (E2 only) shards from all FITS; manifest; commit the code (approval)                                                                                                                                                                                                                           | —                            |
| 21:00 → 21:35 | Final runs: E2 (6), E1 (3 more), E3, error analysis, system measures                                                                                                                                                                                                                                                           | Stop launching runs at 21:35 |
| 21:35 → 21:50 | Execute the notebook with outputs, commit results, push (approval)                                                                                                                                                                                                                                                             | Pushed by 21:50              |
| 21:50 → 22:00 | Work report (section 9)                                                                                                                                                                                                                                                                                                        | **22:00**                    |

**Must:** download, v2 split + audit, frozen label rule + counts, padded conversion + inspection + manifest,
persistence baseline, E2, E1, E3, error analysis, system measures, executed notebook, pushed commit, work report.
**Should:** clean-subset analysis and R9 input comparison (5.3), bucket upload of `processed/v2/`.
**Could (only if every Must is done):** random forest reference row (3.2).

**Fallbacks:** if the export fails, use the 1,260 existing images (569 val cases after the gap) and say so. If
runs are slow, drop to 2 seeds before dropping an experiment. If time runs out, report what ran and list the rest
under "not done". The milestone accepts a documented blocker.

## 1. Decisions (resolved with Adam)

| Topic      | Decision                                                                                                                                                        | Reason                                                                                                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Candidates | Two ImageNet-pretrained models, fine-tuned: **ResNet-18** (`timm/resnet18.a1_in1k`) and **DeiT-Tiny/16** (`timm/deit_tiny_patch16_224.fb_in1k`, `img_size=128`) | Instructor's advice: compare good pretrained models instead of a logistic-regression baseline                                                                                           |
| Baseline   | **Persistence** (deterministic, proposal PR1), plus climatology (TSS = 0 by definition)                                                                         | Required by Milestone 4 ("compare against the baseline") and by PR1                                                                                                                     |
| Image rule | **Zero-pad to square, then resize to 128×128** becomes the v2 rule; stretch survives only as E2's comparison arm                                                | Milestone 3 section 6.7 and feedback item 3                                                                                                                                             |
| Test set   | Untouched. All of Milestone 4 uses the gap-corrected 2015 validation set; the test runs once in Section 9.1 later                                               | Template puts final target comparisons in 9.1                                                                                                                                           |
| Compute    | **Laptop (`mps`)**. Vertex AI is deferred to Milestone 5                                                                                                        | Measured 1.1 s per epoch; no Vertex setup exists (GPU quota, container, data staging), and a failed setup tonight would cost the milestone. Record this in the report as a change to D2 |
| Tracking   | JSON run records plus git commits (no W&B)                                                                                                                      | Already working; record it as a change to the T1 evidence                                                                                                                               |

## 2. Data v2 (closes feedback items 1–3)

`processed/v1` stays frozen. The new rules go into `configs/data.yaml` as a v2 rule set (layout is your choice),
and the outputs go to `processed/v2/`.

### 2.1 Achieved 27-day gap (feedback 1)

Milestone 3 claimed that "the training loader removes the first 27 days of 2015 and 2016". **The first-exploration
code never did** (no gap logic in `src/solarsentinel/models/`). The calendar rule would not have been enough
anyway: seven train HARPs that cross the boundary keep observations until 2015-01-09 06:00.

- **Rule:** drop every observation of a later block whose `slot_time` is earlier than (last `slot_time` of the
  previous block + 27 days).
- **Expected results** (from a quick check; recompute and confirm):
  - val starts 2015-02-05 06:00 → 5,185 observations, 208 positives, 172 HARPs (652 removed, 25 of them positive);
  - test starts 2016-01-28 12:00 → 5,101 observations, 74 positives (376 removed, 3 positive);
  - achieved gaps exactly 27.0 days on both sides.
- Write `processed/v2/observations.parquet`, derived deterministically from v1 by this rule plus a fingerprint,
  and **`processed/v2/split_audit.json`**:
  - counts per split (observations, positives, negatives, flagged negatives, `quality_nonzero`, HARPs);
  - HARPs shared across splits (must be 0);
  - achieved minimum gaps in days;
  - rows removed by the gap, per split.
- **One loader for every modeling entry point** (training, evaluation, persistence, calibration, notebook). For
  example, `solarsentinel.models.dataset.load_observations("v2")` reads v2 and asserts the audit. Add a small pytest
  that fails if any entry point sees a val row before 2015-02-05 06:00.
- Effect on the first exploration: 61 of its 630 val cases were inside the gap. Report the first-exploration
  numbers recomputed on the 569 that remain (no retraining needed: filter the saved `_val_preds.parquet`), so the
  scoring change is explained _before_ new results are compared with old ones.

### 2.2 Frozen label rule (feedback 2)

Freeze this rule and use it for every comparison:

| Use                             | Rule                                                                                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Training                        | Exclude flagged negatives (`has_unattributed_mx & label == 0`), as Milestone 3 planned; keep `quality_nonzero` rows (0.6% of train) |
| Evaluation, primary             | All gap-corrected val cases that have an image                                                                                      |
| Evaluation, secondary ("clean") | Primary minus flagged negatives and `quality_nonzero` rows                                                                          |

- Report positives, negatives and flagged/quality counts for every comparison, in one table.
- First-exploration evidence for the training rule: Run C changed TSS by −0.015 (95% CI −0.068 to +0.035).
- **What the clean subset leaves out (Should):** compare flagged vs clean negatives in train+val on inputs (median
  USFLUX, AREA_ACR, TOTUSJH; share in high-activity months). If flagged negatives are the "hard" active-period
  negatives, the clean score is optimistic: say by how much on val.

### 2.3 Padded cutouts: verify before expanding (feedback 3)

- **Rule:** zero-pad the FITS array to a square (centered; zero field is physically neutral) → the existing
  NaN-fill and anti-aliased resize → 128×128 float16. Record `pad_fraction` per cutout next to `nan_fraction`.
- **Inspect on a small batch first:** the 10 thinnest regions (largest `orig_nx/orig_ny`) and every cutout with
  `nan_fraction > 0`. Save a figure of original vs stretch vs pad, and report:
  - how many pixels the short side gets after padding (for example, a 4:1 region gets about 32 rows);
  - that NaN-filled pixels and padding are both 0 and indistinguishable to the model (a known limitation; record
    it, don't fix it tonight).
- **Distinct version and manifest:**
  - `processed/v2/cutouts/` (padded shards + index);
  - `processed/v2/manifest.json` (sha256 per file, row counts, config hash, code commit).
- **E2 only:** `processed/v2/experiments/m4/cutouts_stretch/` = the v1 stretch rule applied to the same FITS, so
  both arms see identical images.
- Upload `processed/v2/` after approval (Should).

### 2.4 Tonight's download (`--targets`, `--stop-at 20:45`)

At the measured ~1,240 cutouts per hour, 18:40–20:45 gives about 2,600. Priority order:

1. The **19 val positives** (after the gap) not yet downloaded, then about 38 val negatives (after the gap, from
   all negatives) to keep 1:2.
2. Interleave `[TP, TN, TN]`: all **876 remaining train positives**, and train negatives drawn **only from clean
   negatives** (the training rule excludes flagged ones, so downloading them is wasted time).

- Use the same seed in `configs/experiments.yaml`. Files already on disk are skipped.
- Expected if it completes: val about 626 cases (208 positives); train about 3,250 (all 1,086 positives).
- Keep every JSOC limit. Log the throughput.

## 3. Baseline and reference

### 3.1 Persistence (Must)

Predict "flare" if the same HARP had an attributed ≥M1.0 flare peaking in (t − 24 h, t]. That is the label logic
mirrored backwards, using the same attribution and the same flux scale.

- Evaluate it on (a) the exact val cases of E1/E2 and (b) the full gap-corrected val (5,185).
- Sanity check: a quick approximation (each HARP's label at t − 24 h) gave TSS ≈ 0.30–0.34. The exact number will
  differ a little.

### 3.2 Random forest on the 18 SHARP parameters (Could)

Class-weighted, trained on clean train rows, evaluated on the same val cases and the full val. A reference row
only, not a candidate (instructor's advice). It matters because Milestone 3's R4 trigger is defined against an RF.
If it is skipped, the report says so.

## 4. Experiments

Each experiment answers one question and moves one variable. **Pre-register the decision rules below in the
notebook before running.**

**Shared recipe:**

| Setting                            | Value                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --- | ----------------------------------------- |
| Changes from the first exploration | v2 data and gap, frozen label rule, **20 epochs**; everything else the same                                                                      |
| Input                              | `clip(Br, ±c)/c`, `c` = 99.5th percentile of `                                                                                                   | Br  | ` over **train** pixels, computed per arm |
| Augmentation                       | horizontal flip, vertical flip, polarity flip                                                                                                    |
| Loss                               | `pos_weight = n_neg/n_pos` of the training set                                                                                                   |
| Optimizer                          | AdamW, lr 1e-4, weight decay 0.05, cosine schedule, 1 warmup epoch                                                                               |
| Batch                              | 32, fp32 on `mps`                                                                                                                                |
| Selection                          | evaluate the final epoch; no early stopping (selecting on val twice would inflate it; this is a change from Milestone 3's R4 control, so log it) |
| Seeds                              | 42, 0, 1                                                                                                                                         |

**Every run record** adds `git_commit` + `dirty` flag, the dataset version + manifest hash, the label rule, and the
per-arm `c`.

| ID                 | Question                                                                                             | What changes                                            | Held fixed                                                   | Judged by                                                                                                                                                                                                                                                                | Decision rule                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **E2** (run first) | Does keeping the regions' true shape (pad) beat stretching?                                          | Cutout rule: stretch vs pad                             | FITS, split, label rule, ResNet-18, recipe, seeds, val cases | Mean val TSS ± sd over 3 seeds; paired HARP-bootstrap Δ (pad − stretch); ROC-AUC; **TSS and recall on thin regions** (`orig_nx/orig_ny ≥ 2`)                                                                                                                             | Adopt pad unless clearly worse (Δ mean TSS < −0.02 _and_ CI excludes 0)                                                                                 |
| **E1**             | Which pretrained model should the system use, and does it beat persistence?                          | Architecture: ResNet-18 vs DeiT-Tiny (on the E2 winner) | Data, recipe, seeds, val cases                               | As E2, plus PR-AUC\*, latency, size; persistence row on the same cases                                                                                                                                                                                                   | If Δ CI includes 0: choose by calibration result (E3), then size/latency against Section 2.3, then implementation complexity. Record which rule decided |
| **E3**             | Can the selected model's scores be shown as probabilities, or must the dashboard rank (Section 4.3)? | Calibration: none vs Platt vs isotonic                  | Selected model, seed-42 scores, val cases                    | Milestone 3 PR2 checks on 2015 (diagnostic): (1) mean p / observed rate in [0.5, 2]; (2) calibration-slope 95% CI contains 1; (3) 5 equal-count bins, observed-frequency CI contains mean p, at most one miss. Plus weighted Brier skill score and a reliability diagram | All three pass → probability mode; else ranking mode. Either way, report the effect on the interface                                                    |

**E3 mechanics:**

- **HARP-grouped 2-fold cross-fitting within val:** fit on one half of the regions, predict the other, swap. That
  gives out-of-fold calibrated p for every case without fitting and scoring on the same rows.
- **Weight to natural prevalence:** positives `208 / n_pos_downloaded`, negatives `4,977 / n_neg_downloaded`
  (4,977 = gap-corrected val negatives). Update `val_negatives_total`.
- Monotone calibration leaves the ranking unchanged; say so, and re-pick the threshold on the calibrated scale.

## 5. Evaluation, errors, system measures

### 5.1 Metrics

- **Primary: TSS.** Show the TSS curve over thresholds, the TSS-optimal threshold, and a 95% CI from resampling
  whole HARPs (1,000 reps).
- **Also:** ROC-AUC; prevalence-corrected precision and PR-AUC\*; recall and false-alarm rate; confusion matrix.
- **Report n, positives, and distinct flaring HARPs** for every number.
- **Task-level view for the user:** at the chosen threshold and natural 2015 prevalence, how many alerts per day,
  what share are hits, and missed flares per month.

### 5.2 Error analysis (selected model, seed 42)

- **Tag every error** at the threshold with rule-based categories and count each:
  - false negatives by flare class (M1.0–1.9, M2–4.9, ≥M5);
  - false positives where the region produced a C-class flare vs stayed quiet (from `label_class`);
  - flagged negative;
  - multi-NOAA HARP;
  - thin region (aspect ≥ 2);
  - `nan_fraction > 0`;
  - |LON_FWT| > 45°;
  - persistence agreement.
- **Inspect 15–20 cases:**
  - the 5 most confident FNs and the 5 most confident FPs;
  - 5–10 from the largest category;
  - each with its image (pad), p (raw and calibrated), HARP, NOAA_AR, slot time, label_class, flags, USFLUX,
    aspect ratio.
- **Confidently wrong** = calibrated p > 0.9 for an FP, or < 0.05 for an FN (if E3 is not done, use raw scores and
  say so). List every such case.
- **Investigate the most consequential pattern**, likely the missed flares (the costlier error for an operator):
  - recall by flare-class bin and by USFLUX tercile;
  - false-alarm rate for C-flaring vs quiet negatives.

### 5.3 System measures

- CPU latency at batch 1 and batch 15 (median of 200);
- model size in MB;
- seconds per epoch and peak memory on `mps`;
- tonight's export throughput and the projection for the full set;
- cost $0.

Compare each against Section 2.3/2.6: under 50 MB, milliseconds per region on CPU, under 10 min per epoch.

### 5.4 R9 input comparison (Should; inputs only, no test labels)

For 2016–17, compare `quality_nonzero` vs clean records on SHARP-parameter medians (% difference) and simple image
statistics. Also count quality flags by month and number of distinct HARPs. This answers the feedback question on
interpreting a test set that is mostly flagged, without looking at test outcomes.

## 6. Outputs

- `milestone4_experiments.ipynb` (repo root), executed with outputs. Sections:
  1. setup
  2. v2 data and audit
  3. label-rule counts
  4. padding inspection
  5. baseline
  6. E2
  7. E1
  8. E3
  9. error analysis
  10. system measures
  11. summary tables

  Leave `training_experiments.ipynb` unchanged.

- Run records: `experiments/results/m4_<exp>_<model>_<arm>_s<seed>.json` + `_val_preds.parquet`.
- **Figures for the Word document** (PNG, 150 dpi, readable at half-page width) in `docs/figures/m4/`:
  1. experiment comparison (TSS ± CI per arm, persistence line);
  2. original vs stretch vs pad;
  3. reliability diagram, raw vs calibrated;
  4. error gallery;
  5. recall by flare class and false alarms by negative type.
- Commits in logical pieces. **Results commit = the commit cited in the milestone.** Push after approval.

## 7. Don'ts

- Don't touch replay/test 2016–17 outcomes. The R9 input comparison uses inputs only.
- Don't use accuracy, a 0.5 threshold, ImageNet mean/std, or HF/timm image transforms.
- Don't overwrite `processed/v1` or change v1 rules.
- Don't raise JSOC limits or start an export without approval.
- Don't let a Should or Could item delay the work report.

## 8. What the milestone document will need (the work report must supply it)

- **Revisions table:** evidence for each Milestone 3 feedback item (2.1, 2.2, 2.3), the "keep thinking about"
  point (region-level uncertainty), and the "one question" (5.4).
- **Section 7 (7.1, 7.2, 7.3, 7.4, 7.5, 7.7):** from section 4 and 5.
- **Rows to update in place:**
  - 2.2 FR2, PR1, PR2;
  - 2.3 and 2.7 D2 (compute), D3 (input representation);
  - 3.3 T1, T2, T4;
  - 4.3 (probability vs ranking mode);
  - 5.3 R1, R3, R4 (now pretrained models vs persistence; RF no longer the comparison), R6, R9;
  - 6.4, 6.5, 6.6, 6.7.

## 9. Work report (Must, by 22:00)

Write `notes/work-reports/2026-11-02_milestone4.md` following the `CLAUDE.md` template (sections 1–9). Then **add
section 10, "Milestone 4 evidence"**, written so someone who wasn't there can draft the milestone from it alone.
Every number has its source (file, run ID, notebook cell). Anything unverified is marked.

1. **Feedback evidence:**
   - gap rule and audit numbers;
   - label-rule counts table;
   - padding inspection findings, manifest path and hash;
   - status of the parameter-only path (persistence works without images);
   - first-exploration numbers before and after the gap.
2. **Experiment table (7.2):** E1–E3 with question, change, held fixed, judged by, decision rule, and the outcome
   of each rule.
3. **Configuration table (7.3):** run ID, model + HF repo + timm version, dataset version + manifest hash, recipe,
   seed, `c`, n train/val (positives), commit, notebook path.
4. **Results table (7.5):** each arm with mean ± sd over seeds, paired Δ with CI, ROC-AUC, PR-AUC\*, latency, size,
   s/epoch; persistence (and RF, if done) on the same cases and on the full val.
5. **E3 outcome:** the three PR2 checks (pass/fail with numbers), BSS raw vs calibrated, and the resulting
   dashboard mode.
6. **Error analysis:** category counts, the inspected cases as a table, every confidently wrong case, and the
   investigated pattern with numbers.
7. **System measures** vs the targets.
8. **Changes from the Milestone 3 plan, each with its reason:**
   - compute (laptop, Vertex deferred);
   - no W&B;
   - fixed epochs instead of early stopping;
   - flips + polarity instead of rotations;
   - negative subsampling because of download throughput;
   - RF replaced by pretrained comparisons per the instructor;
   - stretch → pad.
9. **Links:**
   - GitHub URLs at the results commit (repo, notebook, `configs/`, `src/solarsentinel/models/`, `split_audit.json`);
   - the figure paths.
10. **Limitations and not done.**

Tell Adam the moment the report is written, with its path.
