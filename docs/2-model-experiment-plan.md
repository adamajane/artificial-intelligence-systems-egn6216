# SolarSentinel: First Model Exploration plan (brief for Claude Code)

_Context brief for the EGN 6216 homework "First Model Exploration" (prep for Milestone 4). It plays the same role
for this step that `docs/data-pipeline-plan.md` played for the data step: what to build, the decisions already
made and why, and the rules that keep the evaluation honest. Read it fully before planning._

**How to work (Adam's standing rules from the data step):**

- Plan first, show the plan, wait for approval. Decisions with real trade-offs are Adam's.
- Ask before starting any JSOC export (a failed request can block the account for about an hour).
- Ask before every commit. Never push without being asked.
- Re-runnable code. Every rule and setting lives in a config file, not in code.
- Use `uv` (`uv add ...`), Python 3.13. No credentials in the repo; `JSOC_EMAIL` stays in the untracked `.env`.

**Read first:** `README.md`, `configs/data.yaml`, `docs/data-pipeline-plan.md`, `notes/decisions-explained.md`,
`playground.ipynb` (its bucket-loading code is the starting point), `src/solarsentinel/data/cutouts.py`, and the
assignment in `notes/first-model-exploration.md`.

---

## 0. Deadline mode: about 3 hours to submission

**First action: ask Adam for the deadline clock time.** Turn the relative times below into absolute ones and put
them at the top of the notebook. Time, not polish, is the constraint. Every step below has a hard stop, and
**partial results with an honest note beat a late submission**. The assignment explicitly accepts a documented
blocker.

### 0.1 Decisions (resolved with Adam)

| Topic | Decision | Why |
| --- | --- | --- |
| Models | ResNet-18 vs DeiT-Tiny, both ImageNet-pretrained via `timm` (section 2.2) | Professor asked for pretrained models compared with each other |
| Aspect ratio | Keep the v1 stretch to 128×128 | No extra work; identical inputs for both models; documented limitation |
| Compute today | **Adam's laptop (Apple Silicon, PyTorch `mps`; CPU fallback)** | About 700 training images train in about a minute per run; Claude Code can run the notebook and save outputs itself; no Colab session or upload step. Colab T4 is for later, bigger runs. Code picks `cuda → mps → cpu`, so the same notebook runs in Colab |
| Images | **Time-boxed stratified download** (0.3); not the 8.4 h subset or the 25.6 h full run | Only 139 pilot cutouts exist, all train |
| Test set | Replay 2016–17 untouched | Assignment rule |

### 0.2 Timeline (T0 = Claude Code starts)

| Time | Work | Hard stop |
| --- | --- | --- |
| T0 → T+0:20 | Build the target list (0.3), add the subset mode to `cutouts.py`, `--dry-run` (free keyword check). **Adam approves → start the export in a separate terminal** | Export running by T+0:25 |
| T+0:20 → T+1:20 | **While it downloads:** `uv add` deps, Dataset/model/metrics code, smoke test on the 139 pilot cutouts, CPU latency and model size (random tensors suffice), notebook skeleton with the Step 1 text and the data-handling notes | **No new export request after T+1:20**; the one in flight may finish |
| T+1:20 → T+1:30 | Build shards from whatever arrived, upload the subset prefix, print counts per stratum | — |
| T+1:30 → T+2:00 | Runs A and B, evaluation, error analysis, Run C | Start writing at T+2:00 whatever state the runs are in |
| T+2:00 → T+2:30 | Fill the table and the submission note (0.8), execute the notebook top to bottom with outputs, commit (Adam approves), push | Pushed by T+2:30 |
| T+2:30 → deadline | Buffer; Adam submits | — |

Extra seeds (0.5) only if the runs are done before T+1:50.

### 0.3 Time-boxed image download

The pilot measured about 2.5 s per cutout (25.6 h for 36,365), dominated by JSOC queue waits of about 76 s per
35-record request. **One hour ≈ 1,400 cutouts**, fewer if daytime queues are longer or older data has to be
staged from tape.

**Target list** (`processed/v1/experiments/fme/targets.parquet`, seed in `configs/experiments.yaml`):

- Four strata from `observations.parquet`: val positives (VP, all 233), val negatives (VN, sample 466 of 5,604),
  train positives (TP, sample 233 of 1,086), train negatives (TN, sample 466 of 23,965). Sample negatives from
  **all** negatives, flagged ones included. Shuffle each stratum with the seed.
- **Interleave in the repeating pattern `VP, VN, VN, TP, TN, TN`** (233 rounds = 1,398 targets). JSOC batches
  follow this order, so **any prefix that finishes is a stratified sample** with about 1:2 positives to negatives
  in both splits. Append a second tier (more TP and TN in the same 1:2 ratio) in case the download runs faster
  than expected.
- Columns: `HARPNUM, T_REC, split, label, stratum, priority, shard` (`shard` = UTC month of `t_rec_utc`, as
  `load_targets` computes it).

**Code changes to `cutouts.py`** (small; reuse `fetch`, `build_shards`, `write_index` and every limit unchanged):

- `--targets PATH`: use that list instead of the pilot window, keeping its order (do not re-sort by `T_REC`
  before `fetch`, or the prefix property is lost).
- `--stop-at HH:MM` (local time): `fetch` starts no new export request after it. Log what is left undone.
- Output paths, so nothing overwrites the pilot shard (`build_shards` rewrites a whole month, and the pilot owns
  `2014-10`):
  - raw FITS → `data/raw/jsoc/sharp_cea_720s_Br/full/<HARPNUM>/...` (the future step-6 root, so the full run later
    skips these files)
  - shards + index → `processed/v1/experiments/fme/cutouts/`
- `--upload`: upload `processed/v1/experiments/fme/` (≈ 45 MB). Upload the raw FITS root and its retrieval log
  only if there is time; it never blocks the homework.
- Leave `max_recordset_chars: 1400` and the one-pending-export rule alone. **Do not experiment with export methods
  today.** If a request fails with "too long" or a pending-request status, stop and go to fallback F1 (0.9). Do not
  retry in a loop.

### 0.4 What the subset means for evaluation

The validation set is negative-subsampled (about 1:2 instead of 1:24). State this plainly in the notebook:

- **Unaffected by the subsampling (primary):** TSS, recall (POD), false-alarm rate (POFD), ROC-AUC. Each is
  computed within one class, so the class ratio doesn't change its expected value. TSS was picked for exactly
  this property.
- **Need correcting:** precision, PR-AUC and the Brier skill score depend on prevalence. Compute them with sample
  weights: each val negative counts `w = 5,604 / n_val_neg_downloaded`, positives count 1. Label them
  "prevalence-corrected". `sklearn.metrics.average_precision_score(..., sample_weight=w)` supports this.
- n is small (≈ 700 val cases, ≈ 233 positives), so **HARP-bootstrap intervals are mandatory** and the A vs B
  difference may be inconclusive. That is a legitimate finding.

### 0.5 Recipe for today (A and B identical except the architecture)

| Setting | Value |
| --- | --- |
| Data | `processed/v1` observations + `experiments/fme` cutouts; train and val strata from 0.3; n per stratum recorded |
| Input | 1×128×128, `clip(Br, ±c)/c` with `c` = 99.5th percentile of `|Br|` over **train** pixels (section 2.4) |
| Augmentation (train) | random horizontal flip, vertical flip, polarity flip (`x → −x`), each p = 0.5 |
| Models | `timm.create_model(name, pretrained=True, in_chans=1, num_classes=1)`, plus `img_size=128` for DeiT |
| Loss | `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)` of the training set used (≈ 2) |
| Optimizer | AdamW, lr 1e-4, weight decay 0.05, cosine schedule, 1 warmup epoch |
| Budget | 15 epochs, batch 32, fp32 (no AMP on `mps`), all layers trainable |
| Selection | evaluate the **final** epoch; per-epoch train and val loss are logged for curves only |
| Seed | 42 (torch, numpy, random). If time allows: seeds 0 and 1 too, reporting mean ± sd |
| Record | torch and timm versions, device, HF repo IDs, `c`, n per stratum, wall time per epoch |

Write one JSON per run to `experiments/results/<run_id>.json` (config, metrics, timings) and the val predictions
to `experiments/results/<run_id>_val_preds.parquet` (`HARPNUM, T_REC, label, p`). Both are small and committed.

### 0.6 Error analysis, constraints, Run C

- **Errors (Step 5):** for the better of A and B, show the 5 most confident false positives and the 5 most
  confident false negatives. For each: image, p, `HARPNUM`, `NOAA_AR`, `slot_time`, `label_class`,
  `has_unattributed_mx`, `multi_noaa`, `USFLUX`, `LON_FWT`, aspect ratio `orig_nx/orig_ny`, `nan_fraction`.
  Write 3–5 sentences on what they share. Also report the overlap between A's and B's error sets.
- **Constraints (Step 5):** see 2.6 for the targets. Measure:
  - CPU latency on the laptop: batch 1 and batch 15 (one live timestep), `torch.inference_mode()`, 20 warm-ups,
    then the median of 200;
  - saved `state_dict` size in MB;
  - training seconds per epoch on `mps`;
  - **download throughput** (cutouts per hour from `exports.json`, the real bottleneck) and the projection for
    the full set.
  - Cost: $0.
- **Run C (Step 6): one change, same val cases, same metrics.** Choose it from the A and B results with this rule,
  and state the expected effect before running:
  1. Learning curves show overfitting (val loss rising while train loss falls) → **freeze the pretrained
     backbone** and train only the head (backbone in `eval()` so BatchNorm statistics stay fixed). Tests how far
     ImageNet features transfer to magnetograms.
  2. Else, if DeiT clearly trails ResNet → **DeiT at 224** (bilinear upsampling, pretrained 14×14 grid) vs. 128
     with resampled position embeddings.
  3. Else → **exclude `has_unattributed_mx` negatives from training** (val unchanged).

  Default if unclear: option 1. Apply it to the better model; to both if minutes remain.

### 0.7 Notebook `training_experiments.ipynb` (repo root)

Sections: 0 header (deadline, run summary, links) → 1 role of AI, candidates, non-AI stages, stage explored first →
2 setup (versions, seed, device) → 3 data (load from bucket, join, split asserts, class balance, handling notes,
normalization fitted on train) → 4 model configuration → 5 smoke test on the pilot → 6 training A and B →
7 evaluation (section 4) → 8 errors → 9 constraints → 10 Run C → 11 experiment table, findings, blocker, next steps.

Keep it thin: logic lives in `src/solarsentinel/models/` (`dataset.py`, `nets.py`, `metrics.py`, `train.py`)
and the experiment config in `configs/experiments.yaml`. **Execute top to bottom and save with outputs:**
`uv run jupyter nbconvert --to notebook --execute --inplace training_experiments.ipynb`. Add a Colab badge like
`playground.ipynb` has.

### 0.8 Submission package

Write `notes/first-model-exploration-submission.md` with the four items Adam pastes into the submission. Fill
the brackets from the run records. Leave the commit ID for Adam (it is known only after the commit).

```markdown
**Notebook:** https://github.com/adamajane/artificial-intelligence-systems-egn6216/blob/<commit>/training_experiments.ipynb (commit `<commit>`)

**Stage explored first:** the per-region flare classifier (proposal C5). It is the only learned stage; the
deterministic stages (sampling, filters, attribution, labelling, split) were verified by the data step's
known-value checks.

**Note.** Selected candidates: ImageNet-pretrained ResNet-18 and DeiT-Tiny, fine-tuned on 128×128 Br cutouts with
an identical recipe, to compare a CNN with a vision transformer. Initial result on [n] validation cases from 2015:
TSS [A] vs [B] (95% HARP-bootstrap CI [..]). Observed failure/limitation: [pattern from the error analysis].
Targeted change: [C], because [reason]; effect: [TSS before → after]. Main limitation: JSOC export throughput
(~[x] cutouts/h) limited this run to [n_train] training and [n_val] validation images, with validation negatives
subsampled (TSS is unaffected; precision and PR-AUC prevalence-corrected). Next action: the 8.4 h subset download
(train positives + 5,000 negatives + full 2015 val), then rerun A/B/C on the full validation set.
```

| Run | Model / configuration | Change from baseline | Val TSS (95% CI) · ROC-AUC · PR-AUC* | Dev cases (n, pos) | Runtime / cost |
| --- | --- | --- | --- | --- | --- |
| A | ResNet-18, recipe 0.5, seed 42 | — (baseline) | | | s/epoch on mps · CPU ms/region · MB · $0 |
| B | DeiT-Tiny/16 @128, recipe 0.5, seed 42 | Architecture only | | same | |
| C | [better model] + [change] | [one change] | | same | |

\* prevalence-corrected (0.4). Also report the TSS-optimal threshold for each run.

### 0.9 Fallbacks

- **F1, export fails or is blocked:** stop exporting. Do the smoke test on the 139 pilot cutouts (end to end,
  outputs shown) and measure the constraints. If any val images arrived, evaluate on them. Document the blocker
  (error text, what was tried) and the next action (rerun the export after the block lifts). This satisfies the
  "documented blocker" path.
- **F2, fewer images than planned:** use what arrived and report n honestly. Below about 50 val positives, call
  the results preliminary in every sentence that cites them.
- **F3, `mps` errors:** fall back to CPU (the data is tiny), and record it.
- **F4, out of time for Run C:** submit A and B, and describe Run C as the planned next run with its expected
  effect.

---

## 1. The assignment in one screen

Six steps:

1. **Role of AI and candidates.** Two or three sentences on what goes in and what comes out; candidates; which to
   run first and why; non-AI stages.
2. **Load and inspect data.** Shapes, labels, class balance; document what was filled, dropped or left unchanged;
   use the Section 6.5 split; fit any learned transformation on train only.
3. **Run the first candidate minimally.** End to end first (a forward pass proves the code runs, not that the
   model is good). Record the configuration before tuning.
4. **Evaluate on validation** with a metric that fits the purpose, plus the number of examples.
5. **Inspect 5–10 errors** with specific examples, including confidently wrong ones. **Measure one practical
   constraint** against the proposal's targets, stating the conditions.
6. **One targeted change**, rerun on the same validation cases with the same metric, compare.

**Submit:** repo link plus commit ID; short note; experiment table; which stage was explored first. **Before
submitting, check:** role of AI and candidates; one approach end to end with initial results; specific outputs
inspected; one change tried or a blocker documented; test data unused; configurations documented; notebook outputs
visible and commit pushed.

## 2. Background decisions

### 2.1 What the AI does (Step 1 text, to adapt)

> The AI component takes one 128×128 SHARP radial-field (Br) magnetogram cutout of a NOAA-numbered active region
> at a 6-hour slot, and outputs the probability that this region produces a flare of class ≥M1.0 (GOES science
> scale) within the next 24 hours. Everything around it is deterministic: sampling, filtering, flare-to-region
> attribution, labelling, splitting, input normalization, the alert threshold and the dashboard.

The alert is a **policy on top of the probability**: a threshold chosen on validation (max TSS). The model is not
retrained to change it.

**Non-AI stages:** 6-h slot sampling with ±60 min tolerance and QUALITY preference, NOAA/±60° filters, flare
attribution (methods a–d), 24-h labelling, chronological HARP-grouped split, Br normalization (a fitted constant),
threshold selection, alert rule.

### 2.2 Candidates

The comparison is **a CNN vs. a vision transformer with matched ImageNet accuracy**. Same data, same preprocessing,
same recipe, so the difference between them comes from architecture (local convolutional bias vs. global attention).

| ID | Model | Source (HF Hub) | Params | License |
| --- | --- | --- | --- | --- |
| A | ResNet-18 | `timm/resnet18.a1_in1k` | 11.7 M | Apache-2.0 |
| B | DeiT-Tiny/16 | `timm/deit_tiny_patch16_224.fb_in1k` | 5.7 M | Apache-2.0 |

Both reach roughly low-70s % ImageNet top-1, and both are small enough for the runtime bundle target (2.6). Add
both sources and licenses to the README provenance section.

Named in Step 1 but not run: the ResNet-50 vs. DeiT-Small scale-up pair (about 90–100 MB each, over the 50 MB
target); EfficientNet-B0 and ConvNeXt-Atto; a hybrid of a CNN embedding plus the 18 SHARP parameters; NASA-IBM
Surya (`nasa-ibm-ai4science/Surya-1.0`), which is full-disk and multi-channel (a thesis direction); feature-based
models on SHARP parameters (parked per the professor).

### 2.3 Why a ViT needs adapting (and how)

The cutouts are float16 `[N, 128, 128]` arrays of raw signed Br in Gauss (NaNs filled with 0 before resizing),
sharded by month, plus `index.parquet`. The default Hugging Face path breaks in four ways:

1. **Not an HF image dataset.** Write a custom PyTorch `Dataset` joining `index.parquet` ↔ `observations.parquet`
   on `(HARPNUM, T_REC)`. (Every model.)
2. **Physical values, not pixels.** **Never use an HF/timm image processor or transform**: they would divide by
   255 and apply ImageNet mean/std. Apply our own normalization (2.4). Cast float16 → float32 on load. (Every model.)
3. **One channel.** Use `in_chans=1`; timm sums the pretrained RGB weights of the first conv/patch embedding.
   (Every model.)
4. **128 vs. 224 (ViT only).** The learned position embeddings are for a 14×14 patch grid. `img_size=128` makes
   timm resample them to 8×8 (64 tokens). Upsampling to 224 is Run C option 2.

Beyond the format: ViTs are data-hungry and lack a CNN's bias toward local structure. That is the hypothesis the
comparison tests, and why DeiT (data-efficient training) was picked.

### 2.4 Input preparation

- Normalization fitted on train only: `c` = 99.5th percentile of `|Br|` over training-image pixels, logged.
  Input = `clip(Br, −c, c)/c`. **Keep the sign** (polarity and polarity inversion lines matter).
- Augmentation as in 0.5. No arbitrary rotations, no brightness or contrast jitter, no random crops.
- Aspect ratio: v1 stretches regions to 128×128 (original width/height 1.3–4.0; `orig_nx` and `orig_ny` are in the
  index). Kept by decision. Mention it as a limitation; padding is a v2 question.

### 2.5 Class imbalance: where it is handled

**The data pipeline deliberately does not rebalance.** It measures prevalence per split, keeps the natural ratio,
and flags uncertain negatives (`has_unattributed_mx`). The proposal handles imbalance at training time; this
homework implements that.

| Split (science scale) | Observations | Positives | Rate | Flagged negatives |
| --- | ---: | ---: | ---: | ---: |
| train 2011–14 | 25,051 | 1,086 | 4.3% | 2,896 |
| val 2015 | 5,837 | 233 | 4.0% | 659 |
| replay 2016–17 (untouched) | 5,477 | 77 | 1.4% | 100 |

Rules:

1. Weighted loss, `pos_weight = n_neg/n_pos` of the training set used. One mechanism per run.
2. Never use a 0.5 threshold. Sweep on val, pick max TSS, report the threshold.
3. Never report accuracy (all-negative scores 96% on the natural split).
4. The probabilities are not calibrated after weighting or subsampling. Show a reliability diagram (weighted as
   in 0.4). Calibration belongs to a later milestone, with proposal target PR2 (Brier skill score > 0).
5. `has_unattributed_mx` is a label-noise flag, **never an input** (it uses future information).
6. The val subsampling in 0.3 is an evaluation shortcut that only works because the primary metrics ignore class
   ratio (0.4). The full-val rerun is the next action.

### 2.6 Targets to compare against (Step 5)

From Adam's Milestone 2 system-design document. **Adam confirms against the submitted version:**
- Section 2.1 "Performance targets": PR1 test TSS ≥ 0.5 and above persistence (final test only, not this
  homework); PR2 Brier skill score > 0; $0 required spend.
- Section 2.3 "Inference and runtime hardware": CPU-only laptop in Docker, milliseconds per 128×128 region, about
  15 regions per timestep, model bundle < 50 MB.
- Section 2.6 and decision D2: reconsider if an epoch exceeds 10 min.

Expected: ResNet-18 ≈ 45 MB fp32 (tight), DeiT-Tiny ≈ 22 MB. Measure, don't assume.

## 3. Data access

- Read anonymously from `gs://solarsentinel_ai-systems-egn6216` with `gcsfs` (`token="anon"`), as
  `playground.ipynb` does, so the notebook also runs in Colab.
- Index columns: `HARPNUM, T_REC, shard, index, orig_ny, orig_nx, nan_fraction`. Inspect the observations schema
  (48 columns) instead of assuming names. It includes `label`, `split`, `slot_time`, `has_unattributed_mx`,
  `multi_noaa`, `quality_nonzero`, `label_class`, `NOAA_AR`, `LON_FWT` and the 18 SHARP parameters.
- **Step 2 checks in the notebook:**
  - shapes, dtype and value range;
  - class balance of the rows actually used;
  - **leakage asserts**: train and val HARP sets are disjoint; no label or flag column is an input; `c` is from
    train only;
  - handling notes: NaN pixels were 0-filled by the pipeline (count val images with `nan_fraction > 0`); targets
    without a file are dropped and counted by stratum; `quality_nonzero` rows are kept; missing SHARP values
    (5–53 rows) are irrelevant to image models.

## 4. Evaluation (all on the val subset; report n and the number of distinct HARPs)

- **Primary: TSS** = TP/(TP+FN) − FP/(FP+TN). Show the TSS-vs-threshold curve, max TSS and its threshold. Max TSS
  picked on the same val set is optimistic: say so.
- Also: ROC-AUC; prevalence-corrected PR-AUC and precision (0.4); confusion matrix; recall and false-alarm rate at
  the chosen threshold.
- **HARP bootstrap:** resample val HARPs with replacement, 1,000 reps, fixed seed; 95% CI for TSS (at the chosen
  threshold) and ROC-AUC.
- Secondary view: metrics with flagged val negatives excluded.
- Learning curves (train and val loss per epoch) and a weighted reliability diagram.

## 5. After the deadline (next actions, not today)

1. **8.4 h subset download** (overnight on the laptop, under the 24 h laptop rule): all 1,086 train positives,
   5,000 train negatives, the **full** 2015 val (5,837 rows). Reuse `--targets`; files already downloaded today
   are skipped.
2. Rerun A, B and C on the full val at natural prevalence (no reweighting needed), with 3 seeds each. Colab T4 or
   Vertex AI if epochs get long.
3. Full step-6 download (about 25.6 h → Compute Engine VM per the plan rule) before Milestone 4's final model.
4. Later: calibration (PR2), persistence baseline, padding vs. stretch (v2), hybrid image + SHARP model.

## 6. Code layout

```
configs/experiments.yaml                  # seeds, strata sizes, recipe, model IDs (data rules stay in data.yaml)
src/solarsentinel/models/{dataset,nets,metrics,train}.py
training_experiments.ipynb                # thin, executed with outputs
experiments/results/<run_id>.json         # config + metrics + timings (committed)
experiments/results/<run_id>_val_preds.parquet
notes/first-model-exploration-submission.md
```

Weights stay out of git (`.gitignore`). Add `torch`, `timm`, `scikit-learn` with `uv add`, then re-export
`requirements.txt`. Suggested commit message: `First model exploration: ResNet-18 vs DeiT-Tiny on a stratified
cutout subset`.

## 7. Don'ts

- Don't touch replay (2016–17).
- Don't use accuracy, a 0.5 threshold, ImageNet mean/std, or HF/timm image transforms.
- Don't overwrite `processed/v1/cutouts/` or change the rules in `configs/data.yaml`.
- Don't raise JSOC limits, change the export method, or start an export without Adam's go-ahead.
- Don't let polish eat the hard stops in 0.2.
- Don't commit or push without asking.
