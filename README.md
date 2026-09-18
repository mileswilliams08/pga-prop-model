# PGA Tour Prop Bet Model

A model for pricing PGA Tour prop bets — birdies-or-better, greens in
regulation, fairways hit, and total strokes — using player season stats
adjusted for the course being played, then compared against sportsbook
odds to find edge.

## How it works (the short version)

1. **Scrape** player stats from pgatour.com (GIR%, driving accuracy,
   birdie rate, scoring average, strokes gained).
2. **Adjust for course**: a player's season GIR% isn't what matters —
   what matters is their GIR% *at this course*, given how hard the
   course plays relative to tour average. We shift each stat up or down
   based on the course's difficulty.
3. **Model the prop**: each prop becomes a probability distribution
   (Binomial for GIR/fairways/birdies, Normal for total strokes), which
   gives you a real number like "there's a 54.6% chance this player gets
   4+ birdies."
4. **Compare to the sportsbook's line**: convert their odds to an
   implied probability, subtract it from your model's probability — the
   difference is your edge.

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
| `blend_years.py` | Optionally blends multiple seasons' stats into one weighted rating per player |
| `features.py` | Cleans stats, applies course-difficulty adjustment |
| `prop_models.py` | Converts adjusted stats into prop probabilities |
| `ev_calculator.py` | Compares your probability to sportsbook-style odds, sizes bets |
| `prizepicks_scraper.py` / `underdog_scraper.py` | Pull current prop lines from those platforms |
| `match_props.py` | Matches platform lines to your model's players by name |
| `update_data.py` | Runs the full pipeline and writes `docs/data/props.json` for the website (this is what the scheduled GitHub Action runs) |
| `main.py` | Runs the pipeline locally and prints results to the terminal — good for testing changes |

## Multi-year blending (using more than one season's data)

By default, the model rates each player using only the current season's
stats. Early in a season — or for a player who's missed events — that's
a small, noisy sample. `blend_years.py` blends multiple years into one
weighted rating per player, favoring recent seasons but still letting
older ones smooth things out:

```
most recent season : 60% weight
one year back        : 25% weight
two years back        : 15% weight
```

A rookie or anyone missing from an older season isn't penalized — their
rating just comes from whichever years they actually have data for.

**To turn it on:**

- **Running locally via `main.py`**: open `main.py` and set
  `USE_BLENDED_YEARS = True` near the top. The first run scrapes all
  three years (takes a few minutes) and caches the result to
  `data/blended_stats.csv`; later runs just re-read that cache. Delete
  the cache file (or change `YEARS_TO_BLEND`) to force a fresh scrape.
- **Running via the automated website** (`update_data.py` /
  GitHub Actions): set `"use_blended_years": true` in `config.json`,
  and adjust `"blend_years"` if you want different years than the
  default `[2025, 2024, 2023]`.

**Trade-off to know about:** blending triples the scraping work (one
ESPN page load per year instead of one), so each run takes noticeably
longer. If you're running this via the scheduled GitHub Action, the
workflow's timeout is already set generously (20 minutes) to cover
this — but it's still one more thing that can go wrong on any given
run. If a single year's scrape fails, that year is just dropped from
the blend (with a printed warning) rather than failing the whole run.

**What this is not**: this blends a player's *general* skill level
across seasons — it's not the same as knowing how a player has
historically performed *at a specific course*, which is a stronger
signal for props but needs different (per-tournament) historical data.
See "Things worth tuning as you go" below.

## Things worth tuning as you go

- **`phi` (overdispersion) in `prop_models.py`**: controls how "streaky"
  a round is assumed to be. Start at 1.15-1.2; once you have a season of
  your own results, you can fit this properly.
- **`std_dev` for total strokes**: set to 2.8 as a reasonable default.
  Real round-to-round scoring variance differs by player (some are more
  consistent than others) — worth calculating per-player once you have
  round-level history.
- **Course profile numbers**: right now you plug these in by hand
  (`course_profile_example()` in `main.py`, or `config.json`'s
  `"course"` block for the website). For a real system, build a small
  database of course-level historical averages (GIR%, birdie rate,
  scoring average by tournament) so this fills in automatically.
- **Course-specific player history**: a stronger signal than general
  multi-year blending (above) would be "how has this player scored at
  *this* course historically" — some players just fit certain courses.
  This needs per-tournament historical results rather than season
  totals, which is a bigger scraping project than what exists today —
  worth treating as a phase 2 once the current pipeline is solid.

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

Edit `config.json` with that week's course numbers (GIR%, driving
accuracy, scoring average, birdie rate for the course — you'll need to
find or estimate these; historical tournament recaps and Data Golf's
free course-fit pages are good sources) and push the change. The next
scheduled run picks it up.

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

- **PrizePicks** has a genuinely public, unauthenticated JSON API
  (`api.prizepicks.com`) — this is the same one many open-source DFS
  tools use. It's stable as these things go, but still unofficial and
  undocumented, so field names could shift without notice.
- **Underdog publishes no API at all.** `underdog_scraper.py` hits their
  app's internal endpoint, which is reverse-engineered from watching
  their web app's network traffic. This is meaningfully more fragile —
  when (not if) it breaks, open your browser's DevTools → Network tab
  while browsing Underdog's golf board, find the current request their
  own app makes, and update the URL/fields in the script to match.
- If either scraper fails, `update_data.py` just leaves those players'
  `platform_lines` empty rather than crashing the whole run — you'll see
  "no line found" on the site instead of a broken page.

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
