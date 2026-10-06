# SolarSentinel

Forecasts, for each solar active region, whether it will produce a flare of class M1.0 or above in the next 24 hours.
It uses SDO/HMI SHARP magnetic-field data and GOES X-ray flare records. This is the final project for the UF course
EGN 6216 Artificial Intelligence Systems.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamajane/artificial-intelligence-systems-egn6216/blob/main/playground.ipynb)

## Setup

Python **3.13** (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`requirements.txt` is exported from `uv.lock`
(`uv export --format requirements-txt --no-hashes -o requirements.txt`) for tools that need pip.

## Run the notebook

`playground.ipynb` is a first look at the dataset: provenance, size, dtypes, example rows, example cutouts,
data-quality checks and written findings. It reads everything anonymously from the public GCS bucket below, so it
needs no credentials and no local data.

- **Locally:** open it in VS Code and pick the project's `.venv` kernel, or run it headless:
  ```bash
  uv run jupyter nbconvert --to notebook --execute --inplace playground.ipynb
  ```
- **Colab:** use the badge above. The first code cell installs only the packages the notebook imports
  (`gcsfs`, `pandas`, `pyarrow`, `numpy`, `matplotlib`).

## Data provenance and licenses

| Source                                                                            | Version                                                                                               | URL                                                                                                                                                     | Retrieved (UTC) | License / credit                                                                               |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| GOES-15 XRS science-quality flare summary (NOAA NCEI): flare class, used as label | `v2-3-0`, yearly netCDF files 2011–2017                                                               | [ncei.noaa.gov/…/xrsf-l2-flsum_science](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-flsum_science/) | 2026-10-06      | "These data may be redistributed and used without restriction." (file metadata)                |
| GOES XRS operational flare event reports (NOAA NGDC): NOAA region numbers         | yearly text reports 2011–2016; 2015 = `_modifiedreplacedmissingrows`; 2017 = `-ytd` (ends 2017-06-28) | [ngdc.noaa.gov/…/x-rays/goes/xrs](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/)                  | 2026-10-06      | US Government work, public domain                                                              |
| SWPC daily solar event reports (NOAA SWPC): region numbers for 2017-06-29 → 12-31 | daily `YYYYMMDDevents.txt`                                                                            | [ngdc.noaa.gov/…/solar_event_reports](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/)                     | 2026-10-06      | US Government work, public domain                                                              |
| SDO/HMI SHARP keywords (JSOC): 18 summary parameters per 12-min record            | series `hmi.sharp_cea_720s`, keyword queries with `drms` 0.9.1                                        | [jsoc.stanford.edu](http://jsoc.stanford.edu/)                                                                                                          | 2026-10-06      | NASA open data, free to redistribute. Credit: "Courtesy of NASA/SDO and the HMI science team." |
| SDO/HMI SHARP `Br` cutouts (JSOC): radial field images, pilot week                | series `hmi.sharp_cea_720s`, segment `Br`; JSOC export requests listed in `exports.json`              | [jsoc.stanford.edu](http://jsoc.stanford.edu/)                                                                                                          | 2026-10-06      | NASA open data, free to redistribute. Credit: "Courtesy of NASA/SDO and the HMI science team." |

Every downloaded file has an entry in a retrieval log stored next to it in the bucket. Each entry records the exact
URL or JSOC query, the retrieval time (UTC), the byte count and a sha256:
`raw/goes/retrieval_log.json`, `raw/jsoc/retrieval_log.json` and `raw/jsoc/sharp_cea_720s_Br/pilot/retrieval_log.json`.

Code in this repository: MIT, see [`LICENSE`](LICENSE).

## Bucket layout

`gs://solarsentinel_ai-systems-egn6216` is public-read. Objects are also reachable over HTTPS as
`https://storage.googleapis.com/solarsentinel_ai-systems-egn6216/<path>`. Use path-style URLs only, because the bucket
name contains `_`.

```
gs://solarsentinel_ai-systems-egn6216/
├── raw/                                   # byte-for-byte copies of source files; never edited
│   ├── goes/
│   │   ├── sci_flare_summary/             # sci_xrsf-l2-flsum_g15_yYYYY_v2-3-0.nc (2011–2017)
│   │   ├── ops_event_reports/             # goes-xrs-report_YYYY*.txt (2011–2017-06)
│   │   ├── swpc_event_reports/YYYY/MM/    # YYYYMMDDevents.txt (2017-06-29 → 2017-12-31)
│   │   └── retrieval_log.json
│   └── jsoc/
│       ├── sharp_keywords/                # YYYY-MM.parquet (84 months) + pilot/20141020-20141027.parquet
│       ├── sharp_cea_720s_Br/pilot/       # <HARPNUM>/<T_REC>.Br.fits (pilot week only) + retrieval_log.json
│       └── retrieval_log.json
├── interim/
│   ├── flares.parquet                     # science ↔ operational flare match, 2011–2017
│   └── flares_attributed.parquet          # + attribution_method per flare (region, sub-peak, location)
└── processed/
    └── v1/                                # curated dataset
        ├── observations.parquet           # one row per (HARP, 6-h slot): keywords, label, split, flags
        └── cutouts/
            ├── YYYY-MM.npy                # float16 [N, 128, 128] Br per month
            └── index.parquet              # (HARPNUM, T_REC) → shard, row, original shape, NaN fraction
```

`processed/v1/manifest.json` (sha256 per file, row counts, config hash) and a copy of `configs/data.yaml` are added
when v1 is frozen.

## How the data is built

Sources, label rules, splits, filters and the build order are in
[`docs/data-pipeline-plan.md`](docs/data-pipeline-plan.md). All data rules live in
[`configs/data.yaml`](configs/data.yaml).
