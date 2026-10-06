# Data pipeline plan (course data step + proposal C1–C4)

Implementation brief for the EGN 6216 data step (upload dataset → `playground.ipynb` → document → commit)
and the proposal's pipeline components C1–C4. **Status: plan, nothing implemented yet.**
Facts marked _(verified)_ were checked against the live sources on 2026-10-06.

---

## 1. Definition of done

- [ ] Pilot dataset, then full dataset, in `gs://solarsentinel_ai-systems-egn6216`
- [ ] `playground.ipynb` runs top to bottom on a clean machine and loads data **only** from the bucket
- [ ] Notebook shows dataset size, number of features, dtypes, 5 example rows per table, a grid of example cutouts, and data-quality checks with written findings
- [ ] Source, version, retrieval date and license recorded (notebook + README)
- [ ] No credentials or personal data anywhere in the repo or its history
- [ ] Notebook committed **with outputs visible**; small commits with descriptive messages
- [ ] `requirements.txt` exported from `uv.lock`; Python version noted in README

## 2. Decisions

| Topic             | Decision                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python            | 3.13 via uv (`.python-version`). Colab's default runtime is also 3.13. Proposal 2.4 says 3.11 → update it.                                                                                                                                                                                                                                                                                                  |
| Package           | Rename `artificial_intelligence_systems_egn6216` → `solarsentinel` (src layout), as stated in proposal 2.4.                                                                                                                                                                                                                                                                                                 |
| Notebook kernel   | Local `.venv` in VS Code. Colab ("Open in Colab") is only a clean-machine smoke test.                                                                                                                                                                                                                                                                                                                       |
| Storage           | GCS bucket `solarsentinel_ai-systems-egn6216` in GCP project `ai-systems-egn6216`. Public read (`allUsers` → `roles/storage.objectViewer`). Confirmed 2026-10-06.                                                                                                                                                                                                                                           |
| Compute           | HiPerGator allocation is **not** confirmed, so nothing depends on it. Pilot on the laptop. Full download: laptop if the pilot projects ≤ 24 h, otherwise one small Compute Engine VM (`e2-standard-2`) in the bucket's region. CNN training: Colab GPU (VS Code Colab extension), Kaggle as the second free option. GCP GPUs only if both fail (they need a paid billing account and a GPU quota increase). |
| Auth              | Reads: anonymous (`gcsfs.GCSFileSystem(token="anon")`). Writes: Application Default Credentials from `gcloud auth application-default login`. Never create or commit service-account key files. The bucket name is not a secret and may live in config.                                                                                                                                                     |
| JSOC export email | Env var `JSOC_EMAIL`, loaded from an untracked `.env`.                                                                                                                                                                                                                                                                                                                                                      |
| Data rules        | One `configs/data.yaml` read by every script (proposal 2.4: avoids train/serve skew).                                                                                                                                                                                                                                                                                                                       |
| GCS URLs          | Bucket name contains `_`, so use path-style URLs only: `https://storage.googleapis.com/solarsentinel_ai-systems-egn6216/<path>` (gcsfs does this already).                                                                                                                                                                                                                                                  |

## 3. Sources

### 3.1 SHARP: `hmi.sharp_cea_720s` (JSOC, via `drms`)

- 12-minute cadence. Prime keys `HARPNUM`, `T_REC`. **T_REC is in TAI, not UTC.**
- Keyword (metadata) queries need no account. Exporting the `Br` image segment needs an email registered at JSOC.
- Keys to pull: `HARPNUM, T_REC, NOAA_AR, NOAA_NUM, NOAA_ARS, LON_FWT, LAT_FWT, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, QUALITY` (bounding box is needed for location attribution) plus the SHARP summary parameters (`USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ, MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, SHRGT45, R_VALUE, AREA_ACR`). Confirm the list against the series' keyword list.
- Observed on 2014-10-22 and 2014-10-24 _(verified)_:
  - 6 of 13 HARPs have `NOAA_AR = 0` / `NOAA_ARS = MISSING` → removed by the NOAA-numbered rule.
  - One HARP can contain several NOAA regions: HARP 4678 → `NOAA_ARS = "12187,12191"`.
  - Records are missing at exact 6-h times: for HARP 4698 the 18:00 and 18:12 records do not exist, and hours 06 and 18 have 3 of 5 records on both days. A `[...@6h]` query on 2014-10-24 returned only the 12:00 slot. → **Never sample on exact clock times.** Take the nearest record within ±60 min of each slot.
  - `CRSIZE1/CRSIZE2` came back as "Invalid KeyLink" from the JSON API → read cutout shape from the FITS header instead.
- License: NASA open data, free to redistribute. Credit: "Courtesy of NASA/SDO and the HMI science team."

### 3.2 GOES-15 science-quality flare summary (label: flare class)

- `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-flsum_science/sci_xrsf-l2-flsum_g15_y{YYYY}_v2-3-0.nc` (yearly files 2010–2020) _(verified)_
- netCDF4/HDF5, ~5.5 MB per year. Variables _(verified)_: `time` (seconds since 2000-01-01 12:00 UTC), `status` (EVENT_START / EVENT_PEAK / EVENT_END / POST_EVENT), `xrsb_flux`, `background_flux`, `flare_class`, `integrated_flux`, `flare_id`, `sequential_flare_num`.
- **No location and no NOAA region number.** Region attribution has to come from 3.3.
- The reprocessed data drops the old SWPC 0.7 scaling factor, so classes should come out about 1/0.7 ≈ 1.43× the operational list. **Verify in the pilot** using the 2014-10-24 flare (X3.1 in the operational list).
- License (file metadata) _(verified)_: "These data may be redistributed and used without restriction."

### 3.3 GOES operational event list (region attribution)

- `https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{YYYY}.txt`. Check which years exist up to 2017. If 2016–17 are missing, fall back to the SWPC event lists.
- Fixed-width text. Example _(verified)_:
  `31777141019  0417 0548 0503 S10E58   ...   X 11    G15  3.9E-01 12192 141023.5`
  → 2014-10-19, start 04:17, end 05:48, peak 05:03, location S10E58, class X1.1, GOES-15, integrated flux, NOAA region 12192.
- 2014 _(verified)_: 2,258 flares; 205 M + 16 X. **75 of 221 M/X flares (34%) have no region number** (e.g. X1.2 on 2014-01-07, X3.1 on 2014-10-24 from AR 12192). Some have a location but no region (X2.2 on 2014-06-10 at S15E80). 2 exact duplicate lines. `N00E00` appears as a placeholder location (e.g. M1.2 on 2014-02-24) → treat as missing.
- License: US Government work, public domain.

## 4. Label logic (C3)

1. Science flares: rows with `status == EVENT_PEAK` give peak time, class and peak flux.
2. Match each to the operational list by peak time (nearest within ±5 min; tune this and report how many stay unmatched). Take the NOAA region and location when present.
3. Label for observation (HARP, t) = max science-quality class among flares attributed to any region in its `NOAA_ARS`, with peak in (t, t + 24 h]. Binary target: ≥ M1.0.
4. Flag, don't silently drop:
   - `has_unattributed_mx`: an ≥M flare with no region peaked anywhere on disk in (t, t + 24 h].
   - `multi_noaa`: `NOAA_NUM > 1`.
     Before flagging, attribute as many flares as possible (all in v1; record `attribution_method` per flare):
     a. NOAA region from the operational list.
     b. Sub-peaks: a science-only flare whose peak falls inside an operational flare's start–end window inherits that flare's region.
     c. Location: operational location (not `N00E00`) inside a HARP's bounding box (`LAT_MIN/MAX`, `LON_MIN/MAX` at the nearest record, small margin), else nearest flux-weighted centre (`LAT_FWT`, `LON_FWT`) within a tolerance set in config.
     d. Everything else stays unattributed.
     Then v1 removes `has_unattributed_mx` rows from the negatives and reports the counts.
5. Convert T_REC from TAI to UTC with `astropy.time.Time(..., scale="tai").utc` (the offset is 34–37 s over 2011–2017).

## 5. Sampling, filters, splits (C2, from the proposal)

- Period 2011-01-01 → 2017-12-31. Chronological splits: train 2011–2014, validation/threshold 2015, held-out replay 2016–2017.
- Group by HARP: a region that crosses a split boundary goes entirely to the earlier split. Log every case.
- 6-h slots at 00/06/12/18 UTC, nearest record within ±60 min. Keep `NOAA_AR != 0` and `|LON_FWT| <= 60`. `QUALITY != 0` → flag (decide later whether to drop).
- Cutouts: `Br` segment → resample to 128×128 float16. Keep the original shape in the observations table.

## 6. Code layout

```
configs/data.yaml               # all data rules + bucket name
src/solarsentinel/
  config.py                     # load configs/data.yaml
  storage.py                    # local path <-> gs:// path; anonymous read, ADC write (gcsfs)
  data/goes.py                  # download 3.2 + 3.3 -> raw/goes/...; parse + match -> interim/flares.parquet
  data/sharp_keywords.py        # one keyword query per month -> raw/jsoc/sharp_keywords/YYYY-MM.parquet (skip if it exists)
  data/observations.py          # slots, filters, labels, flags, split -> processed/vN/observations.parquet
  data/cutouts.py               # export Br per month (resumable, keyed by HARP+T_REC), resample -> processed/vN/cutouts/YYYY-MM.npy
Makefile                        # targets: pilot, dataset (proposal 2.4)
```

- The local `data/` folder mirrors the bucket layout exactly. Sync with `gcloud storage rsync -r data/<dir> gs://solarsentinel_ai-systems-egn6216/<dir>`.
- Runtime dependencies: `uv add drms astropy numpy pandas pyarrow xarray h5netcdf gcsfs pyyaml scikit-image matplotlib requests python-dotenv`
- Dev dependencies: `uv add --dev nbconvert` (ipykernel is already there).
- SunPy is listed in proposal 2.4 but isn't needed for this step. Leave it out unless a concrete need appears.
- After dependency changes: `uv export --format requirements-txt --no-hashes -o requirements.txt`

## 7. Bucket layout

GCS has no real folders, only name prefixes. Don't create anything in the console; prefixes appear on the first upload.

```
gs://solarsentinel_ai-systems-egn6216/
├── README.md                     # what's here, sources, licenses, retrieval dates
├── raw/                          # byte-for-byte copies of source files; never edited
│   ├── goes/sci_flare_summary/   # sci_xrsf-l2-flsum_g15_yYYYY_v2-3-0.nc
│   ├── goes/ops_event_reports/   # goes-xrs-report_YYYY.txt
│   ├── jsoc/sharp_keywords/      # YYYY-MM.parquet (keyword query results, unmodified)
│   └── jsoc/sharp_cea_720s_Br/pilot/<HARPNUM>/<T_REC>.Br.fits   # pilot only
├── interim/                      # parsed and joined, can be regenerated from raw/
│   └── flares.parquet
└── processed/
    └── v1/                       # curated dataset (C4); frozen once published, new rules -> v2
        ├── manifest.json         # sha256 per file, row counts, config hash, git commit, retrieval dates
        ├── data.yaml             # copy of the rules used to build it
        ├── observations.parquet  # one row per (HARP, T_REC): keywords, label, split, flags, cutout pointer
        └── cutouts/
            ├── YYYY-MM.npy       # float16 [N,128,128] per month
            └── index.parquet     # HARPNUM, T_REC, shard, index, orig shape, nan_fraction (sidecar, so observations stays unchanged while images arrive)
```

Where the full raw FITS set (20–85 GB) lives depends on where the full download runs:

- **Laptop run:** raw FITS stay in the local `data/raw/` cache and are not uploaded. JSOC can always re-export them, and pulling 85 GB back out of GCS would cost ~$0.12/GB.
- **VM run (same region as the bucket):** raw FITS go to `raw/jsoc/sharp_cea_720s_Br/`. Transfer inside one region is free, so a later re-processing (e.g. 256×256, proposal D3 trigger) needs no new JSOC export. The VM processes month by month, uploads, then deletes its local copy, so a small disk is enough.
- VM auth: attach a service account with Storage Object Admin on this bucket only. The VM gets credentials from the metadata server (ADC works automatically). No key files.

## 8. Pilot first

- Window: 2014-10-20 → 2014-10-27 (AR 12192 / HARP 4698, includes the X3.1 on 2014-10-24).
- Record:
  - number of observations, positives, and flag counts
  - Br file sizes and cutout shapes
  - JSOC export wall time → projected full-run time. Decision rule: ≤ 24 h → run the full download on the laptop; > 24 h → Compute Engine VM (replaces the proposal's "> 72 h → HiPerGator" trigger)
  - science vs. operational class for the X3.1 (expect ≈ 1.43×)
  - the X3.1 has no region in the operational list → the HARP 4698 windows before it should be flagged

### Step 4 results: Br export and full-run projection (measured 2026-10-06)

Pilot targets: all 139 rows / 9 HARPs of step 3's draft `processed/v1/pilot/observations.parquet` (26 positive; 104 negatives kept and flagged `has_unattributed_mx`, user decision). After step 3 switched to the nearest `QUALITY == 0` record, 33 T_RECs changed; only those 33 were exported again, and the shard and index follow the current file.

- Export: drms `url` / `fits`, 5 requests of 35/35/35/34/33 records (`JSOC_20261006_001655`, `…_001903`, `…_001910`, `…_001913`, `…_002641`). Submit → ready: **74, 85, 64, 75, 83 s** (mean 76 s). Probe requests with 1–2 records took 55–90 s, so the wait is a fixed cost per request, not per record.
- Download: 172 files, 147.7 MB in 89.5 s = **1.65 MB/s** (one connection, files in sequence; 1.3–2.6 MB/s per request). Mean file of the 139 current cutouts **0.87 MB** (0.13–2.15 MB).
- Cutouts: original shapes 164×365 to 800×1496 px. `nan_fraction` > 0 in 2 of 139 files (max 4.2%). The plain 128×128 resize stretches non-square cutouts up to 4× (HARP 4678, 374×1496).
- Processing (read FITS, count/fill NaN, anti-aliased resize, float16): **≈ 20 ms per file**.
- JSOC export limits found (none of them is documented in drms):
  - One pending export per user (status 7) → requests run strictly one after another.
  - Comma-joined record sets pass a keyword query but fail in export processing (status 4).
  - A record-set string of 4,080 chars failed with "Record-set specification is too long", and the failed request (`JSOC_20261006_001687`) then blocked every new export of the account for **≈ 60 min** (03:41 → 04:41). 1,389 chars worked.
  - → One record set per request, `hmi.sharp_cea_720s[? (harpnum=H and t_rec=S) or … ?]{Br}` (S = T_REC in DRMS seconds since 1977.01.01 TAI), capped at 1,400 chars = 35 pairs, the largest size known to work.

Projection for 2011–2017 (step 3 `observations.parquet` regenerated 07:44 with the `QUALITY == 0` slot rule: **36,365 observations**, 1,206 HARPs):

| Part                                                       |      Hours |
| ---------------------------------------------------------- | ---------: |
| Export queue: 1,036 requests × 76 s                        |       21.9 |
| Download: 20.9 GB at 1.65 MB/s                             |        3.5 |
| Processing: 36,365 × 20 ms                                 |        0.2 |
| **Total (requests and downloads in sequence, as piloted)** | **25.6 h** |

- Raw FITS volume is scaled by bounding-box area: FITS bytes are proportional to cutout pixels (2,850 bytes per deg² of `LAT/LON_MIN/MAX` box in the pilot). The full period averages 0.66× the pilot's box area (October 2014 had unusually large regions), so ≈ 21 GB rather than the 31 GB the pilot's mean file size would give. That is the low end of section 7's 20–85 GB.
- **Decision rule: 25.6 h > 24 h → Compute Engine VM** (26.1 h if the queue wait is scaled per record instead of per request).
- The margin is under two hours, and the time is JSOC's per-request queue, which a VM does not shorten. What a VM adds is an unattended 26-h run and raw FITS next to the bucket (section 7). Two levers would bring the laptop under 24 h; neither is applied yet:
  - Download request N's files while request N+1 queues (still one pending export): ≈ 22.1 h.
  - A record-set cap of 2,040 chars (52 pairs, 710 requests): ≈ 18.7 h in sequence. Only worth trying if JSOC confirms the limit, because a too-long request blocks the account for about an hour.
- Uncertainty: 5 queue-wait samples, taken between 03:30 and 07:50 EDT; JSOC load varies.

## 9. `playground.ipynb` outline

0. Title, purpose, how to run
1. Setup: imports. If running in Colab, `pip install` the requirements. `BUCKET` constant. No auth cell (public bucket).
2. Provenance table: source, version, URL, retrieval date, license
3. Load from GCS: flares, keywords, observations, one cutout shard
4. Size, number of features, dtypes (`shape`, `info()`)
5. Examples: `head()` of each table; grid of ~8 Br cutouts titled HARP / time / label (diverging colormap, symmetric limits)
6. First look: missing values, duplicates, class balance per split, value ranges, longitude distribution, missing 6-h slots, unattributed flares, multi-NOAA HARPs
7. Findings (markdown): include the problems that don't have a fix yet. This feeds report Section 6.4.

Execute and save outputs before every commit:
`uv run jupyter nbconvert --to notebook --execute --inplace playground.ipynb`

## 10. Repo hygiene

- `.gitignore`: `.DS_Store` (currently `.DS_STORE`; that only matches on a case-insensitive disk), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `data/*` with `!data/.gitkeep`.
- Don't strip notebook outputs (they're required).
- README: setup command (`uv sync`), Python version, data provenance table, bucket layout link.

## 11. Proposal updates triggered by this step

- 2.3 and 2.7 D2: HiPerGator allocation not confirmed (D2's own trigger) → CNN training on Colab GPU, Kaggle as second free option; HiPerGator only if the course allocation appears.
- 2.6: pipeline trigger becomes "pilot projects > 24 h on the laptop → Compute Engine VM".
- 2.4: Python 3.11 → 3.13.
- 2.5: add GCS with cost arithmetic. Storage ≈ 3 GB × ~$0.02/GB-month ≈ $0.06/month. Egress ≈ $0.12/GB: a notebook run (~0.1 GB) ≈ $0.01, a full processed pull (2.3 GB) ≈ $0.28. Check the current GCS pricing page; the always-free tier may cover all of this in US regional buckets.
- 2.7: add a row D5 "Dataset storage": public-read GCS bucket for the curated dataset + source tables; raw FITS cached on compute. Reconsider if the monthly bill exceeds $5 or the processed set exceeds 20 GB.
- C3 description: the science-quality flare summary has no region information, so attribution comes from the operational event list (Section 4).

## 12. Build order (one Claude Code session per step, commit at the end of each)

Order rule: cheapest and most certain first; tables before images. This follows proposal 2.6: parameters first, so the notebook, baselines and dashboard don't wait on the image download.

| Step | What                                                                                                                   | Output                                                   | Needs                                     | Status  |
| ---- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------- | ------- |
| 0    | Scaffolding: package rename, dependencies, `configs/data.yaml`, `.gitignore`, `requirements.txt`                       | —                                                        | —                                         | ✅ done |
| 1 ✅ | `config.py`, `storage.py`, `data/goes.py`: download both GOES sources for 2011–2017, parse, match sci ↔ ops, upload    | `raw/goes/...`, `interim/flares.parquet`                 | nothing (no account)                      |         |
| 2    | `data/sharp_keywords.py`: pilot week first (time it), then 2011–2017 month by month, resumable, upload                 | `raw/jsoc/sharp_keywords/YYYY-MM.parquet`                | nothing (keyword queries need no account) |         |
| 3    | `data/observations.py`: slots, filters, labels, flags, splits for the **full period** (no images yet)                  | `processed/v1/observations.parquet`                      | steps 1–2                                 |         |
| 4    | `data/cutouts.py`: **pilot week only**. Export Br, resample, shard, upload. Measure throughput → laptop-or-VM decision | `processed/v1/cutouts/2014-10.npy`, pilot FITS in `raw/` | `JSOC_EMAIL` in `.env`                    |         |
| 5    | `playground.ipynb` (section 9 outline) + README provenance; execute with nbconvert; test via Open in Colab             | committed notebook with outputs                          | steps 1–4                                 |         |
| 6    | Full cutout download (laptop or VM, per section 8 rule), then `manifest.json` and freeze `processed/v1`                | complete v1                                              | step 4 measurement                        |         |

After step 5 the course data step can be submitted. Step 6 can run in the background afterwards.

### Known values to check against (verified on 2026-10-06)

- Step 1, ops list 2014: 2,258 flare lines, 2 exact duplicates, 205 M + 16 X = 221 M/X, of which 75 have no region number. X1.2 on 2014-01-07 (peak 18:32) and X3.1 on 2014-10-24 (peak 21:41) have no region; X2.2 on 2014-06-10 (peak 11:42) has location S15E80 but no region.
- Step 1, sci ↔ ops: the 2014-10-24 21:41 flare should be about 1.43× X3.1 in the science file (≈ X4.4). Report the actual ratio, plus the median ratio over all matched flares.
- Step 2: 2014-10-24 at 18:24 TAI has 13 HARPs, 6 with `NOAA_AR = 0`. HARP 4698 ↔ NOAA 12192. HARP 4678 ↔ `12187,12191`. HARP 4698 has no records at 18:00 or 18:12 TAI on 2014-10-24.
- Step 3: the observation for HARP 4698 at 2014-10-24 18:00 must not end up as a clean negative (X3.1 at 21:41 with no region → `has_unattributed_mx`). Report positives, negatives, flagged and dropped counts per split.

### Step 1 results (interim/flares.parquet, checked 2026-10-06)
- 20,482 flares 2011–2017: 11,438 matched, 7,733 science-only, 1,311 operational-only (flux rescaled ÷ 0.7). Ops 2015 uses the `_modifiedreplacedmissingrows` file; 2017 H2 comes from SWPC daily reports (4-digit regions + 10000).
- Median science/ops flux ratio 1.431 (≈ 1/0.7, scaling confirmed). 2014-10-24: ops X3.1 → science X4.5, plus a science-only X3.1 sub-peak at 21:20.
- On the science scale 2014 has 390 M/X flares vs 221 in the ops list: 119 matched flares are C-class in the ops list. 169 of the 390 have no region: 122 matched without region, 44 science-only (21 inside an ops flare window with a region → sub-peak rule), 3 ops-only. Without attribution steps b–c, `has_unattributed_mx` would remove a large share of 2014 negatives → step 3 must report this before dropping anything.

### Working rules for each step

- Start the session by reading this file. Use plan mode for steps 3 and 4 (most logic).
- Every script is idempotent: re-running skips work that is already done locally or in the bucket.
- End each step with a short summary containing real numbers (rows, files, sizes, run time, counts that differ from the known values above). Then stop.
- Commit with a descriptive message (course rule: small commits that say what changed).
