# SolarSentinel: notes for Claude Code

SolarSentinel forecasts, for each solar active region, whether it produces a ≥M1.0 flare in the next 24 hours,
from SDO/HMI SHARP magnetograms and GOES flare lists. It is solo project for UF EGN 6216 (AI Systems), so it is built as a full system: data pipeline → model → replay dashboard →
deployment and monitoring.

## Where things are

- `docs/data-pipeline-plan.md`: how the dataset is built (sources, labels, splits, build order).
- `docs/2-model-experiment-plan.md`: **current step** (First Model Exploration). Read it before model work.
- `configs/data.yaml`: every data rule. The single source of truth; code never hard-codes a rule.
- `src/solarsentinel/`: the package (`data/` pipeline steps; `models/` for training and evaluation).
- `playground.ipynb` (data exploration) and `training_experiments.ipynb` (experiments). Both read from the bucket.
- `notes/`: The developer's private notes and work reports. **Gitignored on purpose; never commit it.**
- Bucket `gs://solarsentinel_ai-systems-egn6216` (public read, us-east1): `raw/` → `interim/` → `processed/v1/`.
  The local `data/` mirrors it and is never committed.

## Commands

```bash
uv sync                                                    # environment (Python 3.13, pinned)
uv add <pkg>                                               # dependencies; then re-export requirements.txt:
uv export --format requirements-txt --no-hashes -o requirements.txt
uv run python -m solarsentinel.data.<step> --help          # pipeline steps (goes, sharp_keywords, observations, cutouts)
uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb   # run and save with outputs
```

## Ground rules (every session)

1. **Plan first, then wait for approval.** Decisions with real trade-offs are the developer's. When you pick a
   default, say so and mark it as awaiting confirmation.
2. **Ask before:** any JSOC export, any commit, any push, uploading or overwriting bucket objects, deleting files.
3. **Data rules live only in `configs/data.yaml`.** `processed/v1` is frozen: new rules mean `v2`. `raw/` is
   never edited.
4. **Secrets:** `JSOC_EMAIL` stays in `.env` and is never printed, logged or written to a file. Bucket writes use
   gcloud Application Default Credentials. No key files anywhere.
5. **JSOC limits** in `configs/data.yaml` (`max_recordset_chars`, one pending export) are measured, hard limits.
   Never raise them. A failed oversized request blocks the account for about an hour.
6. **Replay (2016–2017) is the final test set.** Never use it for training, tuning, thresholds or peeking.
7. **Evaluation conventions:** TSS is primary; never report accuracy or use a 0.5 threshold; choose thresholds
   on validation; fit any learned preprocessing on train only; bootstrap confidence intervals over HARPs (snapshots
   of one region are correlated); `has_unattributed_mx` is a label-noise flag, never a model input.
8. **Scripts are re-runnable:** finished work is skipped, interrupted downloads resume.
9. **Numbers come from running code,** not estimates. Check results against the known values in the plans.
10. **Parallel agents** work in worktrees under `.claude/worktrees/`. List any left behind in the work report.

## Writing style (docs, notebooks, reports)

Plain language. Lead with the point. Explain each technical term where it first appears. Every number comes from
an actual output, and its source (file, run ID, cell) is named. Say what is uncertain or unverified. Use tables
for comparisons and short paragraphs elsewhere. No hype.

## End-of-session work report

**When:** When the developer says "wrap up", "work report" or "end of session", or a session that changed files is about to
end. If unsure, offer it.

**Where:** a new file `notes/work-reports/YYYY-MM-DD_<topic>.md` (a second one on the same day gets `_2`).
Never overwrite an earlier report.

**Before writing, verify:** run `git status` and `git log`, list worktrees, and re-check every number you report
against its source. Anything not verified is marked as such. Never write "should work".

**Sections:**

1. **At a glance:** 3–5 lines covering the session goal, where it ended, and the single next step.
2. **What was done, added and changed:** a table of item | files | commit | status. Include deleted or moved files
   and anything uploaded to the bucket.
3. **How the parts fit together:** a short data-flow sketch (text or a mermaid diagram) from inputs to outputs for
   everything touched, showing where the new pieces plug into the existing pipeline and which config keys drive
   them.
4. **Why:** each decision with the alternatives considered, the reason, and who decided (the developer, or a Claude
   default awaiting confirmation).
5. **Results and key numbers:** each with its source. Flag preliminary results (small n, single seed).
6. **Problems hit and fixes,** including dead ends worth not repeating.
7. **Open decisions at the moment.**
8. **To-do for developer:** split into "needed for the next deadline" and "can wait", as checkboxes.
9. **State at the end:** branch, last commit, uncommitted and unpushed changes, worktrees, bucket changes, and any
   still-running jobs with how to check on or resume them.

Keep it scannable: someone who wasn't watching should know in two minutes what changed, why, and what to do next.

## Maintaining this file

Keep it short. Durable rules only; step-specific plans go in `docs/`. When a new standing rule comes up in a
session, propose the CLAUDE.md edit to the developer rather than making it silently. Update the "current step" pointer
above when the active plan changes.
