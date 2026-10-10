# Weekly tournament checklist

Quick reference for keeping the model correct from week to week.
(Everything here is run from the project folder; push with explicit `git add <files>` — never `git add .`.)

## 1. Finish grading the old tournament BEFORE switching

The final round's results are only graded by the *next* daily run (8am CDT / 13:00 UTC).

- Sunday night: leave `config.json` on the old tournament.
- Monday, after the 8am CDT run: open the Results tab and confirm round 4 picks for the old
  tournament are there.
- Only then change the config. Switching earlier means round 4 never gets graded.
- If the config stays on an old tournament for days, the daily run archives meaningless
  post-tournament snapshots (e.g. `bank_of_utah_championship_2026-10-05.json`). They are skipped
  by grading now, but delete them from `data/props_history/` if you notice them.

## 2. Switch the tournament

Edit `config.json` (on GitHub or locally) and push:

- `tournament_name` -> the new event (should match how ESPN and the historical files name it).
- Leave `field` empty unless you want to restrict players.
- Do NOT touch `year`, `blend_years`, `course_history_years`, `player_course_years` mid-season.

## 3. Run the workflow once and read the log

Actions tab -> "Run workflow", then open the run and look for:

- `Using real course history for '<event>' (N real rounds ...)` -> good.
  `No real course history found ... falling back to the hand-typed 'course' numbers` -> the name
  doesn't match the history files; fix the name or fill in the `course` block in config.json.
- `Player-course history: X/Y players ...` -> a low X is normal for a new or rarely played course.
- A small N of real rounds (one edition ~ 300 rounds) means the course numbers are less
  reliable; treat picks with more caution.
- `blend` lines: all of 2026/2025/2024 (or your configured years) should say "players with usable
  stats". If one says "scrape failed", re-run the workflow.

## 4. Lines (PrizePicks)

- Round 1 lines usually appear a day or two before; later rounds' lines the day before or that
  morning (timing varies).
- After each scheduled run, check the site shows platform lines.
- If there are none, the run keeps the previous props (log says "Keeping the existing
  docs/data/props.json") and archives nothing. Re-run the workflow manually once lines are up,
  before tee times.

## 5. Don't run manually mid-round past round 1

Saved GIR/fairways data is diffed between rounds; a manual run while players are partway
through round 2+ can mix a finished round with a partial one. Round 1 is fine (nobody has
started round 2). The scheduled run is safe because it runs between rounds for US events.

## 6. Events outside the US

- Round numbers come from how many rounds the field has finished (not the calendar), so the
  first scheduled update_results log should say `Stamped ... with predicted_round=N`.
- Check the stamp is the round you expect before trusting that day's grading.

## 7. Occasionally

- `tour_avg` in config.json is hand-typed and should match the historical data (see below).
  Recheck at the start of each season, not every week.
- New season: update `year` and shift `blend_years` forward (e.g. `[2027, 2026, 2025]`).

## Pars / Bogeys or Worse (experimental)

- These need the course's par. Optionally add `"course_par": 71` to config.json for the new
  week; if you leave it out, the pipeline reads par from ESPN once a round has been played
  (so before round 1 they are skipped and the log says so).
- They are tracked on the Results tab but left out of the headline "All props" numbers; pick
  "All props incl. experimental" or the individual prop to see them.

## 8. Two-minute weekly check

- Actions run is green.
- Site shows platform lines for the right round.
- Results tab lists the new tournament in the dropdown; graded counts look reasonable.

## What to set `tour_avg.scoring_avg` to

It is NOT a weekly number. It is the baseline the course adjustment is measured against:
a course's effect is `course_scoring_avg - tour_avg.scoring_avg`. It should equal the average
round score across the same historical data the course profiles are built from, so a course's
offset reflects how much harder or easier than an average round it plays.

To compute it for the years in `course_history_years`:

```
python -c "import pandas as pd, json; yrs=json.load(open('config.json'))['course_history_years']; s=pd.concat([pd.read_csv(f'data/historical_rounds/{y}_rounds.csv') for y in yrs])['round_strokes']; s=pd.to_numeric(s,errors='coerce').dropna(); print(len(s), round(s.mean(),2))"
```

Put the second number in `tour_avg.scoring_avg` (it was 70.9 hand-typed vs ~70.3 from the data,
which made courses look ~0.6 strokes easier than they are). Update it only when
`course_history_years` changes or a new season of data is added.
