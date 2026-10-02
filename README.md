# PGA Tour Prop Bet Model

A model for pricing PGA Tour prop bets — birdies-or-better, greens in
regulation, fairways hit, and total strokes — using player season stats
adjusted for the course being played, then compared against sportsbook
odds to find edge.

## How it works (the short version)

1. **Scrape** player stats from ESPN (GIR%, driving accuracy, birdie
   rate, scoring average), blended across multiple seasons (on by
   default — see "Multi-year blending" below) with thin-sample players
   shrunk toward the tour average so a rookie's few rounds don't get
   treated as gospel.
2. **Adjust for course**: a player's season GIR% isn't what matters —
   what matters is their GIR% *at this course*, given how hard the
   course plays relative to tour average. This used to be a number you
   typed in by hand each week; it's now looked up automatically from
   REAL per-round results at that same event in past years (see "Real
   course adjustment" below), falling back to a hand-typed estimate only
   for events with no scraped history yet.
3. **Model the prop**: each prop becomes a probability distribution
   (Beta-Binomial for GIR/fairways/birdies, Normal for total strokes),
   which gives you a real number like "there's a 54.6% chance this
   player gets 4+ birdies."
4. **Compare to the sportsbook's line**: convert their odds to an
   implied probability, subtract it from your model's probability — the
   difference is your edge.

All of the above has been backtested against real per-round PGA Tour
results across two full seasons (2024, 2025), not just assumed to work —
see "Validating the model" below for what was actually tested and what
it found.

## Setup

You'll need Python 3.9+ installed. Then, in a terminal, in this folder:

```bash
pip install playwright pandas numpy scipy
playwright install chromium
```

## Running it

**Step 1 — see the model work with sample data (no scraping needed):**

```bash
python main.py
```

This runs the full pipeline on a few example players so you can see the
shape of the output before dealing with real scraped data.

**Step 2 — scrape real stats:**

```bash
python scraper.py --year 2025 --out data/2025_stats.csv
```

This opens a headless browser, visits each PGA Tour stats page, and
saves a combined CSV. It takes a couple minutes (there's a deliberate
pause between pages to avoid hammering their site).

Important: **run this on your own computer**, not a cloud server — PGA
Tour's site can be stricter with traffic from datacenter IP ranges.

**Step 3 — plug real data into the model:**

Open `main.py` and replace the `sample_data()` call with:

```python
raw = pd.read_csv("data/2025_stats.csv")
```

## Files

| File | What it does |
|---|---|
| `espn_scraper.py` | Pulls player season stats from ESPN's golf stats pages (the active scraper — `scraper.py`, targeting pgatour.com, is kept only as a reference; PGA Tour's bot protection blocks it) |
| `blend_years.py` | Blends multiple seasons' stats into one weighted rating per player, and shrinks thin-coverage players (rookies, anyone missing a season) toward the tour average |
| `features.py` | Cleans stats, applies course-difficulty adjustment, bakes in the birdie-rate bias correction |
| `prop_models.py` | Converts adjusted stats into prop probabilities (Beta-Binomial for GIR/fairways/birdies, Normal for strokes) |
| `ev_calculator.py` | Compares your probability to sportsbook-style odds, sizes bets |
| `prizepicks_scraper.py` / `underdog_scraper.py` | Pull current prop lines from those platforms |
| `match_props.py` | Matches platform lines to your model's players by name |
| `historical_scraper.py` | Scrapes REAL per-round results (not season averages) from PGA Tour's own API — one row per player per round per tournament. Powers both the course-history adjustment and the round-level backtest. Run once per season you want data for (`python historical_scraper.py --year 2025`, ~50 minutes) |
| `course_history.py` | Builds real per-tournament course-difficulty profiles from `historical_scraper.py`'s output, matching tournaments across years by name |
| `player_course_history.py` | Builds real per-(player, tournament) history — this specific player's own track record at this specific event, with sample-size-aware shrinkage toward their course-adjusted general rate. The "horses for courses" signal; see "Course-specific player history (phase 3)" — built but not yet validated |
| `backtest.py` | Tests whether multi-year blending actually beats a naive single-season baseline (it does — see "Validating the model") |
| `backtest_rounds.py` | The real calibration test: do the model's predicted probabilities match what actually happens, against real per-round outcomes? Also where every tuning decision below (shrinkage, birdie bias, course adjustment) was validated, not just guessed |
| `update_data.py` | Runs the full pipeline and writes `docs/data/props.json` for the website (this is what the scheduled GitHub Action runs) |
| `main.py` | Runs the pipeline locally and prints results to the terminal — good for testing changes |

## Multi-year blending (using more than one season's data)

Rating a player on only the current season's stats is noisy, especially
early in a season or for anyone who's missed events. `blend_years.py`
blends multiple years into one weighted rating per player, favoring
recent seasons but still letting older ones smooth things out:

```
most recent season : 60% weight
one year back        : 25% weight
two years back        : 15% weight
```

A rookie or anyone missing from an older season isn't penalized — their
rating just comes from whichever years they actually have data for, but
see the next paragraph for why that alone wasn't enough.

**Shrinkage toward the tour average**: a player who only appears in one
thin season (a rookie, or someone who missed most of two years) still
got a full-confidence rating from that one season alone — their small
sample was being treated as if it were as reliable as a player's full
3-year history. Backtesting confirmed this was hurting calibration, so
`blend_years.py` now pulls thin-coverage players toward the tour average
automatically (`shrink_toward_mean()`, `k=0.3` — tuned and validated in
`backtest_rounds.py --shrink-sweep`, confirmed to still be the best
value after course adjustment was added later). This happens
automatically whenever blending is used; no setup needed.

**To turn blending on:**

- **Running locally via `main.py`**: open `main.py` and set
  `USE_BLENDED_YEARS = True` near the top. The first run scrapes all
  three years (takes a few minutes) and caches the result to
  `data/blended_stats.csv`; later runs just re-read that cache. Delete
  the cache file (or change `YEARS_TO_BLEND`) to force a fresh scrape.
- **Running via the automated website** (`update_data.py` /
  GitHub Actions): `config.json` has `"use_blended_years": true` by
  default, with `"blend_years"` set to the 3 most recent seasons.

**Trade-off to know about:** blending triples the scraping work (one
ESPN page load per year instead of one), so each run takes noticeably
longer. If you're running this via the scheduled GitHub Action, the
workflow's timeout is already set generously (20 minutes) to cover
this — but it's still one more thing that can go wrong on any given
run. If a single year's scrape fails, that year is just dropped from
the blend (with a printed warning) rather than failing the whole run.

## Real course adjustment

Early versions of this model used one hand-typed course-difficulty
estimate per tournament (`config.json`'s `"course"` block) — numbers you
had to research or guess each week. Backtesting found this was by far
the single biggest thing holding the model back: GIR, Birdies-or-better,
and Strokes were all landing at essentially the same accuracy as "just
guess the season-wide average" (see "Validating the model" below),
because a flat line applied to every tournament can't capture the fact
that some courses play much harder than others.

`historical_scraper.py` + `course_history.py` fix this with real data
instead of guesses: once you've scraped a season's real per-round
results, `course_history.py` computes each tournament's actual
field-wide GIR%/driving accuracy/scoring average/birdie rate — the
course's real difficulty, measured, not estimated. `update_data.py`
looks up the upcoming week's `tournament_name` against this history
automatically (matching across years even when a sponsor name changes,
e.g. "... presented by Workday") and uses it instead of the hand-typed
numbers. If no scraped history matches (a new event, or a name that
didn't match), it quietly falls back to `config.json`'s `"course"`
block — nothing breaks, you just lose the benefit for that one event.

**To set this up / grow it over time:**

1. Scrape each season you want course history from:
   `python historical_scraper.py --year 2025` (repeat for other years —
   ~50 minutes each, real API calls, saves progress incrementally).
2. Set `"course_history_years"` in `config.json` to the years you've
   scraped, e.g. `[2025, 2024, 2023]`. More years of real history helps
   (confirmed in `backtest_rounds.py` — going from 1 year to 2 years of
   course history improved every stat further), so scrape another year
   when you get the chance rather than stopping at one.
3. Set `"tournament_name"` each week to that event's real name — this
   now does double duty as both the site's display label and the course
   lookup key.

**What this is not**: this is *course* difficulty (how hard does Augusta
play relative to tour average), not *player-at-this-course* history (how
has Scottie Scheffler specifically performed at Augusta). The latter is
a real, stronger signal some players use — some players just fit certain
courses. See "Course-specific player history (phase 3)" below — it's now
built, just not yet validated against real data.

## Course-specific player history (phase 3)

The course adjustment above is a *main-effects-only* model: take a
player's general skill, shift it by however hard this specific course
plays for anyone. That assumes a hard course is equally hard for every
player, which ignores "horses for courses" — some players genuinely
over- or under-perform their own numbers at one specific venue (shot
shape fit, course familiarity, whatever) for reasons the model doesn't
need to know, only detect.

`player_course_history.py` measures this directly from real per-round
data: for whatever prior years you've scraped, it groups by (player,
tournament) and computes that player's own GIR%/driving
accuracy/scoring average/birdie rate at that *specific* event, with a
sample-size column. The catch, and the reason this isn't just plugged in
by default: with 2-3 years of data, most players have played a given
event only 2-4 times — nowhere near the ~50-150+ rounds backing a
course's field-wide profile. `shrink_player_course_rate()` pulls a
player's course-specific rate back toward their own *course-adjusted*
general rate (not the tour average — their own expected number at this
course), in proportion to how many real rounds back it, so a hot week or
two can't dominate a player's prediction on its own.

**This is built and wired into `backtest_rounds.py`, but not yet proven
to help** — unlike every other change in this README, it hasn't been
validated against real results yet. Do that before trusting it:

```
python backtest_rounds.py --season 2025 --train-years 2024 2023 2022 \
    --player-course-sweep --player-course-years 2024 2023
```

This tries several shrinkage strengths (including "off" — course-adjusted
only, no player-course layer) and reports which one actually lowers the
Brier score. If every candidate loses to "off," this signal isn't adding
anything at current sample sizes and isn't worth turning on. If one wins,
use `--player-course-adjust --player-course-k <that value>` (alongside
`--course-adjust`) to confirm it on the full `backtest_rounds.py` run,
the same way course adjustment itself was validated before being trusted.
Expect smaller gains than course adjustment's, if any — the sample sizes
here are much thinner.

## Validating the model (backtesting)

Two scripts exist specifically to check whether the model's choices
actually help, rather than just sounding reasonable:

- **`backtest.py`**: tests multi-year blending against a naive
  single-season baseline using real ESPN season stats. Confirmed
  blending wins on every stat (5-14% lower error) and that the exact
  blend weights (60/25/15 vs. alternatives) don't meaningfully matter.
- **`backtest_rounds.py`**: the real test — does the model's predicted
  probability for a specific player-round actually match what happens,
  against REAL per-round results from `historical_scraper.py`? This is
  how every tuning decision in this README was actually decided, not
  guessed:

  | Change | Effect (2025 backtest, vs. "always guess the season average") |
  |---|---|
  | Shrinkage (thin-sample players → tour average) | GIR and Strokes flipped from losing to beating the baseline; Fairways improved further |
  | Birdie rate bias correction (+0.01) | Fixed a systematic undercount (ESPN's birdie stat likely excludes eagles) |
  | Real course adjustment | The big one: GIR +11%, Fairways +13%, Birdies +5%, Strokes +8% — by far the largest lever found |

  Also worth knowing: **not every stat has equally strong signal.**
  Before course adjustment was added, Fairways was the only prop with a
  clearly repeatable edge across two separate seasons (2024 and 2025) —
  GIR, Birdies, and Strokes were all within a fraction of a percent of
  "just know the average," meaning the model's confidence on those three
  wasn't as earned as Fairways' was. Course adjustment closed most of
  that gap, but it's still worth re-running `backtest_rounds.py` on a
  fresh season occasionally rather than assuming today's numbers hold
  forever.

  Useful flags: `--debug` (shows match-rate diagnostics and unmatched
  player names), `--phi-sweep` / `--shrink-sweep` / `--birdie-bias-sweep`
  (re-test those specific tuning choices against new data),
  `--course-adjust --course-years 2024 2023` (test course adjustment
  with a specific set of prior years), and `--player-course-sweep` /
  `--player-course-adjust --player-course-k <k>` (test the newer,
  not-yet-validated course-specific player history signal — see
  "Course-specific player history (phase 3)" above).

- **Known data gap, not a bug**: `backtest_rounds.py` typically matches
  ~71% of real player-rounds to a pre-season prediction. The unmatched
  ~29% are mostly Korn Ferry Tour graduates, international/DP World Tour
  players, and limited-status rookies who play in "opposite field"
  events (Corales Puntacana, Bermuda, Myrtle Beach, etc.) but don't have
  a multi-year PGA Tour ESPN stats history — there's genuinely no prior
  data to predict from, not a name-matching defect.

## Things worth tuning as you go

- **`phi` (overdispersion) in `prop_models.py`**: controls how "streaky"
  a round is assumed to be. Backtesting (`--phi-sweep`) found this
  barely matters — candidates from 1.0 to 2.2 all land within ~1% of
  each other on Brier score, with or without course adjustment. Not
  worth spending more time on.
- **`std_dev` for total strokes**: same story as `phi` — tested and
  found not to be a meaningful lever. 2.8 is fine as-is.
- **Course-specific player history**: built (`player_course_history.py`,
  wired into `backtest_rounds.py` via `--player-course-adjust` /
  `--player-course-sweep`) — see the dedicated section above. Still
  needs a real-data validation pass before it's trusted or wired into
  `update_data.py`'s live predictions.
- **Weather (phase 4, deliberately not built yet)**: wind in particular
  has a real, likely large effect on GIR/scoring/birdies, but it's being
  left manual for now rather than automated. Reasoning: this pipeline
  updates weekly, while weather varies round-to-round and even
  wave-to-wave (morning vs. afternoon tee times) within a single round —
  a much finer grain than anything else in this model. It also hasn't
  been backtested (no historical weather data merged into
  `historical_scraper.py`'s output yet), and every other tuning decision
  in this project went through real validation before being trusted —
  guessing a weather fudge factor would repeat the same mistake the
  original hand-typed course numbers made. For now: treat forecasted
  wind/rain as a manual gut-check before locking in a bet (lean toward
  unders on GIR/birdies, overs on strokes, for bubble players on bad-
  weather days). If this gets built properly later, the honest path is:
  pull historical weather per tournament/round (e.g. Visual Crossing's
  historical weather API, keyed to course coordinates + date), merge it
  into the per-round data, and backtest whether high-wind rounds
  actually diverge from the course-adjusted prediction — *then* build
  live forecast integration if that holds up.

## Turning this into a self-updating website (free)

This uses GitHub Actions (scheduled automation) + GitHub Pages (free
hosting) — no server, no database, no monthly cost.

**One-time setup:**

1. Create a new **private** GitHub repo and push this whole folder to it.
2. In the repo, go to **Settings → Pages** → set source to
   **Deploy from a branch**, branch `main`, folder `/docs`. Save.
   GitHub gives you a URL like `https://yourname.github.io/repo-name/`.
3. Go to **Settings → Actions → General → Workflow permissions** and
   select **"Read and write permissions"** — this lets the scheduled
   job commit updated data back to the repo.
4. That's it. The workflow in `.github/workflows/update.yml` runs daily
   (edit the cron line to change the schedule) and updates
   `docs/data/props.json`, which your live page reads automatically.

**Each week before a tournament:**

Set `"tournament_name"` in `config.json` to that week's event (its real
name — this is also the course-history lookup key, see "Real course
adjustment" above) and push the change. If it's an event with scraped
history, course difficulty fills in automatically; if not, it falls back
to the hand-typed `"course"` block, which you can still edit for a new
event with no history yet (historical tournament recaps and Data Golf's
free course-fit pages are good sources for a rough manual estimate). The
next scheduled run picks it up either way.

**Using the site:**

Pick a prop type from the dropdown, see every player's model probability,
type in the sportsbook's American odds for any player when you're
checking a bet, and it instantly shows your edge and expected value —
all calculated in your browser, nothing sent anywhere.

**Want it to trigger on demand instead of waiting for the daily cron?**
Go to the repo's **Actions** tab → select "Update PGA Props" → **Run
workflow**.

## Auto-pulling PrizePicks / Underdog lines (no manual entry)

`prizepicks_scraper.py` and `underdog_scraper.py` pull current lines
directly, and `update_data.py` matches them to your model's players by
name automatically — the website shows the platform's line right next
to your model's probability, no typing required.

**Important differences between the two:**

- **PrizePicks** has a genuinely public, unauthenticated JSON:API board,
  but not at the host you'd guess: `api.prizepicks.com` sits behind
  DataDome bot-management and blocks a plain `requests` call (a 403
  redirecting to a `geo.captcha-delivery.com` challenge).
  `prizepicks_scraper.py` instead hits
  `partner-api.prizepicks.com/projections` — a different, undocumented
  host serving the same live board without that protection, confirmed
  by a real working capture. No browser, no stealth/fingerprint tricks —
  just an honest GET with a normal User-Agent. Note it's specifically
  `partner-api`, not the similarly-named `api.prod01.universe
  .prizepicks.com` (a different host behind the same DataDome
  protection — an early version of this script guessed that one from a
  related project's config default and got blocked). If `partner-api`
  ever starts blocking requests too, the fix is the same kind of fix as
  Underdog's below: capture a fresh working request from DevTools
  (Network tab, filter "projections") and compare against the
  URL/headers in the script.
- **Underdog publishes no API at all.** `underdog_scraper.py` hits their
  app's internal endpoint, which is reverse-engineered from watching
  their web app's network traffic. This is meaningfully more fragile —
  when (not if) it breaks, open your browser's DevTools → Network tab
  while browsing Underdog's golf board, find the current request their
  own app makes, and update the URL/fields in the script to match.
- If either scraper fails, `update_data.py` just leaves those players'
  `platform_lines` empty rather than crashing the whole run — you'll see
  "no line found" on the site instead of a broken page.
- **A clean HTTP 200 with an empty board isn't a scraper failure.**
  Between tournaments (off-weeks, match-play events like the Presidents
  Cup/Ryder Cup), these platforms sometimes post no PGA lines for anyone
  yet — the response comes back valid but with empty `lines`/`players`
  dicts. Re-check once the next tournament's field is set.

**On the site**, since PrizePicks/Underdog pay a fixed multiplier per
pick count rather than per-leg odds, there's a single "breakeven
hit-rate" input instead of per-row odds. Set it to whatever accuracy
your specific entry size (2-pick, 3-pick, Power Play vs. Flex, etc.)
needs to profit — check your app's payout table for that number — and
every row auto-flags as "clears" or "pass."

**Worth knowing:** scraping a betting/DFS platform's live lines sits in
a greyer area than scraping public stats — you're pulling data the
platform generates for its own paying users, not published research.
This is built for personal use with light request volume (once a day),
not high-frequency polling, and you should expect it to occasionally
need maintenance as these platforms change their apps.

## A few honest caveats

- **Sample size matters.** A few weeks of predictions won't tell you if
  this model is actually beating the market — sportsbooks price golf
  props using far more infrastructure than a solo project will match
  out of the gate. Track results before betting real money on it.
- **Scraping and site terms**: check PGA Tour's terms of use before
  running this against their site regularly — this script is built for
  personal, non-commercial use with reasonable rate limiting, not
  high-frequency scraping.
- **Data Golf** (datagolf.com) offers an actual paid API built for this
  exact use case (golf betting models) — it's far more reliable than
  scraping a site not designed to be scraped, and worth considering if
  you want to spend less time maintaining a scraper and more time on
  the model itself.
