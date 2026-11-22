# SolarSentinel: interaction evaluation kit (section 4.4)

_Task script, observation sheets and self-evaluation forms for the Milestone 5 interaction evaluation. Adam runs
the sessions; the observations are his. Claude Code adjusts page names and control locations to the built UI
before Sunday 13:00 (plan section 12) but does not change the tasks or the success criteria._

---

## 1. What this evaluation is for

**Evaluation question (from 4.4, Milestone 2):** can a proxy user, starting cold, find the highest-risk region,
tell whether the data are current and complete, and correctly interpret what a probability and an alert mean,
**without mistaking missing data or NOT ASSESSED regions for all-clear**?

**Method:** three parts.

| Part                  | Who                                  | What                                                                                             |
| --------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Cognitive walkthrough | Adam, alone, before the peer sessions | Step through each task and answer four questions per step (section 9)                            |
| Peer walkthrough      | 3 friends as proxy users (P1–P3)     | Think-aloud on the 4.2 scenario with two injected states (STALE, NOT ASSESSED), then four questions |
| Heuristic evaluation  | Adam, alone                          | Nielsen's 10 heuristics on the built dashboard (section 10)                                      |

Do the cognitive walkthrough **first**. It often finds obvious problems that would otherwise waste a participant.

**Limits, stated in advance (and in 4.4):** the participants are not satellite operators, so this shows whether
people can read the interface, not whether operators would make better decisions (Milestone 3 feedback). With
n = 3, findings are qualitative. They are friends, so they may be polite; the facilitator is the developer, so he
may lead without noticing. Count misreadings, not compliments.

## 2. Success criteria (written before the first session)

The M2 criteria were set for 5 participants (≥ 4 of 5, i.e. 80%). With 3 participants, 80% means **all 3**. Use
that, and report the counts as they are.

| Goal                | Criterion with n = 3                                                                          | Source        |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------- |
| G1 Task success     | All 3 complete the four core tasks (T1–T4 below) unaided                                      | 4.2           |
| G2 Time             | The shift-start check (T1) takes ≤ 2 minutes                                                  | 4.2           |
| G3 Error            | **No participant** reads NO DATA, STALE or NOT ASSESSED as "safe". One misreading triggers R5 | 4.2, R5       |
| G4 Clarity          | All 3 restate P = 25% correctly in natural-frequency terms                                   | 4.2, T2       |
| HR1 Official?       | All 3 say it is not an official forecast                                                      | 2.2 HR1, T3   |
| HR3 Threshold       | Change the threshold, state the trade-off, reset                                              | 2.2 HR3       |

## 3. Before the sessions (checklist)

- [ ] Walkthrough build running; date set to **2017-09-05 12:00 UTC**; bundle ID noted: ____________
- [ ] Scenario switches tested: normal → `stale` → `not_assessed` (and back)
- [ ] Threshold at the default; browser zoom at 100%; other tabs closed
- [ ] Timer (phone) ready; one printed observation sheet (section 7) per participant
- [ ] Cognitive walkthrough (section 9) done, and any blocking problem fixed **before** P1
- [ ] Tasks and wording **frozen**: the same for all three participants
- [ ] Sessions about 20 minutes each, one at a time, so later participants don't overhear the earlier ones

## 3a. Where things are in the built dashboard (added by Claude Code for the build; tasks and criteria unchanged)

_Filled from the build as of 2026-11-20 night; re-checked on the walkthrough build (plan section 12)._

| Thing in the tasks | Where it is |
| --- | --- |
| Pages | Left sidebar, top: **Overview**, **Region detail**, **Tracking**, **About / model card** |
| Banner + SWPC link (T4) | Top of every page: "Research prototype, not an official forecast", link to https://www.swpc.noaa.gov/ |
| Data time and data age (T1) | Chips under the banner: "Replay clock", "Data time", "Data age" (plus mode chip and model bundle ID) |
| Highest-risk region (T1) | Overview, table "Assessed regions, highest chance first", row Rank 1 |
| NOT ASSESSED / NO DATA (T1, T6) | Overview: counts in the summary box under the title, and table "Regions without a forecast" below the region table; disk map at the bottom (cross = NOT ASSESSED) |
| What ALERT means (T2) | Region detail: text under the probability ("An alert is a flag to look closer…", share of alerts followed by a flare); Overview "Alert" column |
| Threshold (T3) | Sidebar, "Alert threshold" slider ("Alert when the chance is at least (%)"); a custom value shows the chip "✎ Custom threshold" and the button **↺ Reset to default**; the table under it shows alerts per day, alerts followed by a flare and flares missed per month on 2015 data |
| Date (setup) | Sidebar "Date (UTC)": **◀ 6 h earlier** / **6 h later ▶**, Day picker, Time picker |
| Scenario switch (T5, T6) | Sidebar, bottom, collapsed expander **Session settings** → Scenario: normal / stale / not_assessed. Or open the URL with `?scenario=stale` or `?scenario=not_assessed` (e.g. `http://127.0.0.1:8501/?scenario=stale`). Switch while the participant is not looking |
| STALE (T5) | Brown box "⏸ STALE DATA. The newest data are from … , N hours old" above the tables; the tables are the last known values |
| NOT ASSESSED scenario (T6) | The regions that would alert are shown as NOT ASSESSED ("near the limb"), so the alert list is empty while regions without a forecast are listed |

## 4. Session script

Read the boxed text roughly as written. Keep your own words neutral.

### 4.1 Consent (verbal, 30 s)

> "Thanks for helping. I'm testing a dashboard for a course project, not you, so there are no wrong answers. If
> something is confusing, that's the dashboard's fault and exactly what I need to know. Taking part is voluntary
> and you can stop at any time. I'll take written notes on what you do and say, and I won't record your name. You'll
> be P1, P2 or P3. Is that OK?"

Record only "yes" on the sheet. No audio or screen recording unless they explicitly agree. (If you are unsure
whether UF expects anything more for a class evaluation with friends, a one-line email to the professor settles it.)

### 4.2 Think-aloud (1 min)

> "While you work, please say out loud what you're looking at, what you expect, and what you're thinking, even
> small things like 'I'm not sure what this means'. I'll mostly stay quiet. If I don't answer a question, that's on
> purpose; I want to see what you'd do on your own."

Practice: ask them to think aloud while finding today's date on their phone's lock screen.

### 4.3 Briefing: the persona and the situation (1 min)

> "Imagine you're Lena, an operations engineer at a small company running 12 small Earth-observation satellites.
> Solar flares can disrupt radio links and GPS. Tomorrow around 12:00 UTC you have a firmware upload planned, and
> you'd rather not do it during a big flare. Every morning you check this dashboard for a couple of minutes before
> deciding whether to go ahead, postpone, or have someone on call. It is now **5 September 2017, 12:00 UTC**."

Do not explain the dashboard. Do not mention STALE or NOT ASSESSED.

### 4.4 Facilitator rules

- Neutral prompts only: "What are you thinking?", "What would you do next?", "What do you expect that to do?"
- Do not answer questions about the interface. Say: "What do you think it means?"
- Stuck for **60 seconds**: give one hint, and mark the task "with help".
- Write down **exact words** for every misreading. Quotes are the strongest evidence in 4.4.

## 5. Tasks

Give each task as written, one at a time. Start the timer when you finish reading T1.

| #   | Scenario state  | What you say                                                                                                                       | Correct / success                                                                                                                                               | Checks         |
| --- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| T1  | Normal          | "Do your morning check. Which region would worry you most, and are today's data current and complete?" **(timed)**              | Names the top-ranked region (AR 12673 on this date, if it ranks first in the build); says the data are current (data time and age) **and** notices any NOT ASSESSED regions or count | G1, G2, F1, F2 |
| T2  | Normal          | "What does the ALERT on that region tell you?"                                                                                     | Probability is above the current threshold; it is a flag to look closer, **not** "it will flare"; mentions or finds what share of alerts were followed by a flare | G1, F3         |
| T3  | Normal          | "Your manager wants fewer false alarms. Change the setting that controls that, tell me what you gain and what you give up, then put it back the way it was." | Raises the threshold; says fewer alerts but more missed flares; resets in one action and the "custom" chip disappears                                         | HR3, F4        |
| T4  | Normal          | "Is this an official forecast? Where would you check before deciding?"                                                            | "No, a research prototype"; points to the NOAA SWPC link                                                                                                        | G1, HR1, F5    |
| T5  | **STALE**       | Switch scenario. "It's the next morning. Do your check again. Is it OK to go ahead with the upload?"                              | Notices that the data are old (STALE, "last known"); does **not** conclude "no alerts, so go ahead"; says they'd check SWPC or wait for fresh data            | **G3**, F1, HR2 |
| T6  | **NOT ASSESSED** | Switch scenario. "A colleague says: 'No alerts today, so we're clear.' Do you agree?"                                             | Disagrees or qualifies: points to the NOT ASSESSED region(s) and says the system can't see them (near or behind the limb)                                      | **G3**, F2, HR2 |

Stop the timer for **G2** when they have answered **both** parts of T1. Write down the time.

## 6. Interpretation questions (after the tasks, dashboard still open)

Ask in this order. Point to a region showing about 25% (or the closest value in the build; write down which value
you used).

| #   | Question                                                                                      | Correct answer looks like                                                                                    | Misreadings to note                                            |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Q1  | "In your own words, what does this 25% mean?"                                                  | "About 1 in 4 similar situations were followed by an M-class or larger flare within 24 hours"               | "Low, so safe"; "a 25%-strength flare"; "it will flare"; "25% of the Sun" | **G4**, F3 |
| Q2  | "If you lowered the threshold, what would happen to alerts and to missed flares?"            | More alerts, more false alarms, fewer missed flares                                                         | Thinks the threshold changes the probability itself           | HR3            |
| Q3  | "What does NOT ASSESSED mean, and what would you do about a region marked like that?"        | The model produced no forecast for it (e.g. too close to the edge of the disk); check another source        | "Not dangerous"; "not important"; "no data, so fine"          | **G3**, F2     |
| Q4  | "How much would you rely on this tool for the upload decision, and why?"                     | Uses it as one input next to the official forecast; mentions limits (prototype, small regions, history)     | Full reliance ("it says no alert, so go"); total dismissal      | F5, T2         |

**Debrief (2 min):** "What was the most confusing moment?" and "If you could change one thing, what would it be?"
Thank them. Then explain what the injected states were, so they don't leave thinking they failed.

## 7. Observation sheet (one per participant; print three)

**Participant:** P__  **Date/time:** ____________  **Consent given:** ☐ yes  **Bundle ID:** ____________
**Background (one line, no names):** e.g. "engineering student, no space-weather knowledge" ______________________

| Task | Success (✔ unaided / H with help / ✘) | Time (T1 only) | Where they hesitated or what they looked at | Misreading (exact words) | Notes |
| ---- | ------------------------------------- | -------------- | ------------------------------------------- | ------------------------ | ----- |
| T1   |                                       |                |                                             |                          |       |
| T2   |                                       |                |                                             |                          |       |
| T3   |                                       |                |                                             |                          |       |
| T4   |                                       |                |                                             |                          |       |
| T5   |                                       |                |                                             |                          |       |
| T6   |                                       |                |                                             |                          |       |

| Question | Their answer (as close to verbatim as you can) | Correct? (✔ / partly / ✘) |
| -------- | ---------------------------------------------- | -------------------------- |
| Q1 (value shown: ___ %) |                               |                            |
| Q2       |                                                |                            |
| Q3       |                                                |                            |
| Q4       |                                                |                            |

**Debrief: most confusing moment:** ______________________________________________
**Debrief: one thing to change:** ______________________________________________

**Problems seen in this session** (one line each; rate severity afterwards with the scale in section 8):

| # | Problem | Where (page/element) | Task | Severity (0–4) |
| - | ------- | -------------------- | ---- | -------------- |
| 1 |         |                      |      |                |
| 2 |         |                      |      |                |
| 3 |         |                      |      |                |

## 8. Summary sheet (fill after all three sessions)

**Severity scale (Nielsen):** 0 not a problem · 1 cosmetic · 2 minor · 3 major (fix before release) · 4 blocks the
task.

| Goal | Criterion (n = 3)               | P1 | P2 | P3 | Met? |
| ---- | ------------------------------- | -- | -- | -- | ---- |
| G1   | T1–T4 unaided                   |    |    |    |      |
| G2   | T1 ≤ 2 min (write the times)    |    |    |    |      |
| G3   | No "safe" reading in T5, T6, Q3 |    |    |    |      |
| G4   | Q1 correct                      |    |    |    |      |
| HR1  | T4 correct                      |    |    |    |      |
| HR3  | T3 done, trade-off stated       |    |    |    |      |

**Issues across participants** (merge duplicates; seen by how many?):

| Issue | Seen by (n of 3) | Severity | Evidence (quote, task) | Outcome: **change** / **confirmation** / **unresolved** | What was done |
| ----- | ---------------- | -------- | ---------------------- | ------------------------------------------------------- | ------------- |
|       |                  |          |                        |                                                         |               |
|       |                  |          |                        |                                                         |               |
|       |                  |          |                        |                                                         |               |

**R5 trigger:** did any participant read NO DATA, STALE or NOT ASSESSED as safe? ☐ no ☐ yes → the state redesign is
the one change Claude Code makes (plan section 12), and it is re-checked.

## 9. Cognitive walkthrough (Adam, before the sessions)

For each step of T1–T4, take the view of a first-time user like Lena and answer the four standard questions. A "no"
is a problem. Write down why.

| Task | Step (action) | 1. Will the user try to do this? | 2. Will they notice the control or information is there? | 3. Will they connect it with what they want? | 4. Will they see that it worked? | Problem / note |
| ---- | ------------- | -------------------------------- | -------------------------------------------------------- | -------------------------------------------- | -------------------------------- | -------------- |
| T1   | Read the top of the region table |                  |                                                          |                                              |                                  |                |
| T1   | Find data time and age           |                  |                                                          |                                              |                                  |                |
| T1   | Notice the NOT ASSESSED count    |                  |                                                          |                                              |                                  |                |
| T2   | Interpret ▲ ALERT                |                  |                                                          |                                              |                                  |                |
| T3   | Find and move the threshold      |                  |                                                          |                                              |                                  |                |
| T3   | Read the operating point         |                  |                                                          |                                              |                                  |                |
| T3   | Reset                            |                  |                                                          |                                              |                                  |                |
| T4   | Find the banner and SWPC link    |                  |                                                          |                                              |                                  |                |

## 10. Heuristic evaluation (Adam, on the built dashboard)

| #   | Heuristic                                    | What to look at here                                                                 | Problem found (or "none") | Severity |
| --- | -------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------- | -------- |
| H1  | Visibility of system status                  | Data time and age, mode chip, STALE / NO DATA / NOT ASSESSED, loading states         |                           |          |
| H2  | Match with the real world                    | Natural-frequency wording, flare classes, UTC times, region names users know (NOAA AR) |                           |          |
| H3  | User control and freedom                     | Threshold reset, date stepping back, leaving region detail                           |                           |          |
| H4  | Consistency and standards                    | Same state names and icons on every page; same number format for p                   |                           |          |
| H5  | Error prevention                             | Can an empty list look like "all clear"? Can a custom threshold be forgotten?        |                           |          |
| H6  | Recognition rather than recall               | Is the meaning of ALERT and the threshold visible where it is used?                  |                           |          |
| H7  | Flexibility and efficiency                   | Sorting, jumping to a date, keyboard use                                             |                           |          |
| H8  | Aesthetic and minimalist design              | Is the morning check visible without scrolling? Anything that competes with it?     |                           |          |
| H9  | Help users recognise and recover from errors | What the user sees when the scorer is down (8.5 deliberate failure)                  |                           |          |
| H10 | Help and documentation                       | About / model card: limits, small-region notice with its sample count                |                           |          |

## 11. How this goes into 4.4

Keep the M2 plan text in 4.4 and add under **Findings**: the method as run (who, when, n, the build's bundle ID);
the summary table from section 8; two or three quotes; **at least one change, confirmation or unresolved issue**,
with what was done; and the limits from section 1. If a finding changes a user assumption, reference it from 4.1,
and add the change to the integration log (plan section 10).
