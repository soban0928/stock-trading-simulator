# Stock Trading Simulator

A day-by-day backtesting engine for US equities, written in Python. It replays
historical daily prices one trading day at a time, runs trading strategies against a
virtual $10,000 portfolio, stores every trade and daily snapshot in SQLite, and
measures performance against an S&P 500 (SPY) benchmark in a Streamlit dashboard.

The emphasis is on **correctness**: exact decimal accounting, no look-ahead bias,
explicit execution timing, and a test suite that checks these properties directly.

> **Educational project.** It uses historical data only, never places real trades,
> and does not manage real money. Backtested results do not predict future returns.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [How it works](#how-it-works)
  - [Backtesting engine](#backtesting-engine)
  - [Portfolio accounting](#portfolio-accounting)
  - [Strategies](#strategies)
  - [Performance analytics](#performance-analytics)
  - [Fair strategy comparison](#fair-strategy-comparison)
  - [SQLite persistence](#sqlite-persistence)
  - [Streamlit dashboard](#streamlit-dashboard)
- [Results](#results)
- [Limitations](#limitations)
- [Testing](#testing)
- [Project structure](#project-structure)

---

## Highlights

- **Day-by-day replay engine** that only exposes prices up to the current day, so
  strategies cannot see the future.
- **Two execution models:** fill at the signal day's close (`same_close`, the
  default) or at the next trading day's close (`next_close`), with pending orders,
  fill-day position sizing, and rejected/unfilled order tracking.
- **Exact portfolio accounting** using `Decimal`: commissions, fractional shares
  (down to 0.000001), average-cost basis and realised profit/loss.
- **Strategies:** SPY buy-and-hold and a 50/200-day simple moving-average
  crossover, registered under unique names.
- **Analytics:** total and annualised return, maximum drawdown, volatility and
  Sharpe ratio, compared with a SPY buy-and-hold benchmark over the same dates.
- **Robustness testing:** every strategy run side by side over the full history and
  over 2015–2019, 2020–2022 and 2023–2026, under identical conditions.
- **SQLite persistence** with transactional, duplicate-safe saves.
- **Interactive Streamlit dashboard** with equity curve, drawdown chart, metrics,
  trade and daily history tables, and the strategy comparison.
- **253 automated tests**, including tests that prove signals cannot use future
  prices and that next-day orders cannot fill early.

**Tech stack:** Python · pandas · SQLite (Python standard library `sqlite3`) ·
Streamlit · Altair · yfinance · pytest

---

## Architecture

```
 Yahoo Finance
      │  simulator/download.py      (yfinance → one CSV per ticker)
      ▼
 data/prices/*.csv                  date, close   (split-adjusted closes)
      │  simulator/data_loader.py   (validates files, aligns trading days)
      ▼
 PriceHistory ──────────────► simulator/engine.py  Simulation
                                │   day-by-day replay, execution timing
                                │   uses simulator/portfolio.py  (cash, positions, trades)
                                │   calls simulator/strategies.py (buy-and-hold, SMA crossover)
                                ▼
                     trades + daily snapshots
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
   simulator/storage.py (SQLite)      simulator/comparison.py (in memory,
                 │                     every strategy × every period)
                 ▼                              │
   simulator/analytics.py  ◄────────────────────┘
   (metrics, SPY benchmark)
                 │
                 ▼
   dashboard/  (Streamlit UI)   and   command-line reports
```

Each layer has a single job. The portfolio enforces trading rules but knows
nothing about dates; the engine controls time and prices but not the rules; the
strategies only decide; analytics only measure; the dashboard only displays
numbers computed by the modules above it.

---

## Getting started

Developed and tested on macOS with Python 3.14.

### 1. Install

```bash
git clone https://github.com/soban0928/stock-trading-simulator.git
cd stock-trading-simulator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `pandas`, `yfinance`, `streamlit` (which
includes Altair) and `pytest`.

### 2. Download price data

Prices are not included in the repository. Download them from Yahoo Finance
(requires an internet connection):

```bash
python -m simulator.download                  # from 2015-01-01 (default)
python -m simulator.download --start 2018-01-01
```

This saves `data/prices/<TICKER>.csv` for 15 stocks (AAPL, MSFT, NVDA, GOOGL,
AMZN, META, JPM, GS, MS, BAC, V, JNJ, KO, WMT, XOM) and the SPY benchmark. The
ticker list, benchmark and start date are set in `simulator/config.py`.

### 3. Run strategies and reports

```bash
python -m simulator.run_strategy --list                    # available strategies
python -m simulator.run_strategy                           # run and save all of them
python -m simulator.run_strategy --execution next_close    # next-day execution
python -m simulator.analytics spy_sma_crossover_50_200     # one run vs SPY
python -m simulator.comparison                             # all strategies × all periods
python -m simulator.storage                                # list saved runs
python -m examples.buy_and_hold                            # small example: 10 AAPL shares
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

It opens at http://localhost:8501. Press `Ctrl+C` in the terminal to stop it.

### 5. Run the tests

```bash
pytest
```

The tests use small hand-written price files, so they need no internet
connection and no downloaded data.

---

## How it works

### Backtesting engine

`simulator/engine.py` — `Simulation`

- Holds a **current day** that only moves forward. `sim.run(strategy)` calls the
  strategy once per trading day, oldest first, then records a `DailySnapshot`
  (cash, market value, total value, profit/loss, holdings).
- Strategies can only see `sim.price(ticker)` (today's close) and
  `sim.past_closes(ticker)` (closes up to and including today). The full price
  table is private, so there is no way to read a later price.
- Optional `start` and `end` dates; indicators may still look back at prices
  before `start` (legitimate history), never after the current day.

**Execution timing** (`Simulation(prices, execution=...)`):

| Setting | Order decided after day *t*'s close fills at | Notes |
|---|---|---|
| `same_close` (default) | day *t*'s close, immediately | Simple, but optimistic: in practice you cannot see a close and still trade at it |
| `next_close` | day *t+1*'s close | Orders are queued with `submit_order()` and filled when the simulation reaches the next trading day |

In `next_close` mode:

- "All cash" and "all shares" orders are **sized on the fill day**, from that day's
  cash, holdings and price, so an overnight price jump cannot overspend.
- **Sells fill before buys** on the same day, so sale proceeds can fund purchases.
- A fixed-quantity order that is unaffordable on the fill day is recorded in
  `rejected_orders` (no commission charged) and the simulation continues.
- An order placed on the last day has no next day and appears in `unfilled_orders`.
- Calling `sim.buy()` / `sim.sell()` directly raises an error, so nothing can fill
  on the signal day by mistake.

The price files contain closing prices only, so **next-day open execution is not
supported** and opening prices are never estimated. Adding it would require
downloading and validating an `open` column and a fresh download of every file.

### Portfolio accounting

`simulator/portfolio.py` — `Portfolio`

- Money and share quantities use **`Decimal`, not `float`**, so cash balances are
  exact. Money is rounded to cents (half-up) at each trade.
- **Commission** (default $1.00) is charged on every buy and sell. The default
  starting cash is $10,000.
- **Fractional shares** down to 6 decimal places. "Buy with all cash" rounds the
  share count *down*, so it never spends more than the cash available.
- **Average-cost basis:** the commission paid on a purchase is included in the
  position's cost; selling records **realised profit/loss** on each trade.
- **Validation:** buying without enough cash raises `InsufficientFundsError`;
  selling more than is held raises `InsufficientSharesError`; malformed tickers,
  quantities, prices or dates raise `InvalidOrderError`. A failed trade changes
  nothing.

### Strategies

`simulator/strategies.py` — each strategy is registered with a unique name, which
is also the name its results are saved under. Duplicate names are rejected.

| Name | Rule |
|---|---|
| `spy_buy_and_hold` | Invest all cash in SPY at the first opportunity (after one commission) and hold to the end |
| `spy_sma_crossover_50_200` | Buy SPY with all cash when the 50-day simple moving average crosses **above** the 200-day; sell all shares when it crosses **below** |

**SMA crossover details**

- Averages are the mean of the last 50 / 200 closes, **including the current day**.
- A crossing compares the current day with the previous trading day: "crosses
  above" means the 50-day was at or below the 200-day yesterday and above it today.
- No signal until both averages exist for today *and* yesterday (201 closes). With
  data starting 2015-01-02, the first possible signal is **2015-10-19**.
- A buy signal while invested, or a sell signal with nothing held, does nothing. If
  the 50-day is already above the 200-day when a run starts, that is not a
  crossing, so the strategy waits in cash for the next upward cross.
- Strategies are stateless: the same data always produces the same trades.

### Performance analytics

`simulator/analytics.py` — metrics are computed from the value series: the starting
cash followed by each day's closing portfolio value.

| Metric | Definition |
|---|---|
| Total return | final value ÷ starting cash − 1 |
| Annualised return (CAGR) | (final ÷ start)^(1 ÷ years) − 1, with years = calendar days ÷ 365.25 |
| Maximum drawdown | largest fall from a previous peak to a later value |
| Volatility | sample standard deviation of daily returns × √252 |
| Sharpe ratio | mean(daily return − daily risk-free rate) ÷ std. dev. of daily returns × √252 |

- The risk-free rate defaults to **0%** (`RISK_FREE_RATE` in `simulator/config.py`)
  and can be changed with `--risk-free-rate`, e.g. `0.04` for 4%.
- Values that cannot be calculated (for example volatility from one day, or Sharpe
  ratio for an all-cash portfolio) are reported as `n/a` instead of a misleading
  number.
- **Benchmark:** SPY buy-and-hold over exactly the strategy's dates, built with the
  same engine and the same `BuyAndHold` strategy. If SPY is missing a date, the
  comparison raises an error instead of silently misaligning.

### Fair strategy comparison

`simulator/comparison.py` runs every registered strategy side by side over several
periods, in memory (nothing is saved):

| Period | Trading days in this data | Earlier days available for indicator warm-up |
|---|---|---|
| Full history | 2015-01-02 → 2026-10-21 | 0 |
| 2015–2019 | 2015-01-02 → 2019-12-31 | 0 |
| 2020–2022 | 2020-01-02 → 2022-12-30 | 1,258 |
| 2023–2026 | 2023-01-03 → 2026-10-21 (data ends here) | 2,014 |

Within a period, every strategy gets the **same trading days, $10,000 starting
cash, $1.00 commission and execution setting**, and starts fresh in cash. The
benchmark is the `spy_buy_and_hold` strategy itself under the same settings, so its
start date matches the other strategies (including under `next_close`).

- **Warm-up:** moving averages may use closes from *before* a period starts (they
  are already in the past), but never after the current day, and nothing after the
  period's end is used. Tests confirm that truncating the data at the period's end,
  or changing every later price, leaves results unchanged.
- Each period also reports trades, the share of days invested, and how the
  crossover's averages stood on the period's first day.

### SQLite persistence

`simulator/storage.py` saves runs to `data/simulator.db` (kept out of Git).

- Four tables: `runs`, `trades`, `snapshots` and `holdings`, linked by foreign keys
  with cascading deletes.
- Money and quantities are stored as `TEXT` so `Decimal` values round-trip exactly.
- Each save is a **single transaction**: a failure part-way leaves nothing behind.
- Run names are unique. Saving an existing name raises `RunExistsError` unless
  `overwrite=True`, which replaces the old copy. The strategy runner uses
  overwrite, so re-running a strategy keeps exactly one copy. Runs using
  `next_close` are saved with a `__next_close` suffix.

```python
from simulator.storage import list_runs, load_snapshots, load_trades, save_simulation

save_simulation(sim, "my_strategy")
trades = load_trades("my_strategy")        # list[Trade]
history = load_snapshots("my_strategy")    # list[DailySnapshot]
```

### Streamlit dashboard

`dashboard/app.py` (layout only) and `dashboard/data.py` (data preparation, no
Streamlit imports). All numbers come from the storage, analytics and comparison
modules; the dashboard does not recalculate them, and it only reads data.

**Saved run view**

- Overview: starting cash, final value, total profit/loss, total return and the
  difference from SPY.
- Interactive equity curve against SPY, with a hover crosshair and tooltip.
- Drawdown chart, risk metrics, and a side-by-side metrics table against SPY.
- Trade history and daily portfolio history tables (daily history downloadable as
  CSV).
- Sidebar: choose a saved run and set the risk-free rate.

**Strategy comparison view**

- Every strategy against SPY for each robustness period, with a same-day / next-day
  execution switch, bar charts of annualised return and max drawdown, and a CSV
  download. Needs only the price files, not the database.

Clear messages are shown when no runs are saved or price data is missing.

---

## Results

**Setup:** daily SPY closes from **2015-01-02 to 2026-10-21** (2,968 trading days),
downloaded from Yahoo Finance in October 2026; $10,000 starting cash; $1.00
commission per trade; risk-free rate 0%. Prices are split-adjusted but **not**
dividend-adjusted, so all returns exclude dividends.

### Full history

| Strategy | Execution | Final value | Total return | Annualised | Max drawdown | Volatility | Sharpe | Trades |
|---|---|---|---|---|---|---|---|---|
| SPY buy-and-hold | same-day close | $37,629.01 | +276.29% | +11.89% | −34.10% | 17.56% | 0.73 | 1 |
| SMA 50/200 crossover | same-day close | $23,271.12 | +132.71% | +7.42% | −34.10% | 14.26% | 0.57 | 11 |
| SPY buy-and-hold | next-day close | $38,321.07 | +283.21% | +12.06% | −34.10% | 17.55% | 0.74 | 1 |
| SMA 50/200 crossover | next-day close | $24,023.57 | +140.24% | +7.71% | −34.10% | 14.26% | 0.59 | 11 |

### By period (same-day close)

| Period | Strategy | Total return | Annualised | Max drawdown | Volatility | Sharpe | Days invested |
|---|---|---|---|---|---|---|---|
| 2015–2019 | SPY buy-and-hold | +56.66% | +9.41% | −20.18% | 13.50% | 0.73 | 100% |
| | SMA 50/200 crossover | +33.35% | +5.93% | −13.77% | 10.03% | 0.63 | 69% |
| 2020–2022 | SPY buy-and-hold | +17.71% | +5.60% | −34.10% | 25.07% | 0.34 | 100% |
| | SMA 50/200 crossover | +32.62% | +9.89% | −12.87% | 11.50% | 0.88 | 56% |
| 2023–2026* | SPY buy-and-hold | +102.99% | +20.49% | −19.00% | 14.88% | 1.33 | 100% |
| | SMA 50/200 crossover | +61.87% | +13.52% | −19.00% | 14.14% | 0.97 | 92% |

\* Partial period, ending with the data on 2026-10-21.

### How to read these results

- Buy-and-hold had the higher total return over the full history and in two of the
  three sub-periods. The crossover generally had lower volatility.
- **The crossover's 2020–2022 result is largely an artefact of the start date.** On
  2020-01-02 its 50-day average was already above the 200-day, so the fresh run
  waited in cash until the next upward cross (2020-07-09) and happened to miss the
  2020 crash. The continuous full-history run was invested through that crash: its
  sell signal came on 2020-03-30, a week after the market bottom, which is why its
  maximum drawdown (−34.10%) matches SPY's.
- In 2015–2019 the crossover could not trade before 2015-10-19 (indicator warm-up),
  while buy-and-hold was invested from day 1.
- Next-day execution changed fill prices and, on this data, slightly improved both
  strategies. That is specific to this history, not a general property.
- These figures come from one market over about 12 years. They are **not evidence
  that either strategy is profitable or robust** in the future. Re-downloading
  prices may change the numbers slightly if Yahoo Finance revises its data.

---

## Limitations

- **Historical backtesting does not predict future performance.** Results describe
  what fixed rules would have done on one past price history.
- **Transaction costs are simplified:** only a flat $1.00 commission per trade. No
  bid/ask spread, slippage, market impact or taxes.
- **Dividends are not modelled.** Prices are split-adjusted but not
  dividend-adjusted, which understates buy-and-hold returns in particular.
- **Cash earns no interest** while a strategy is out of the market.
- **Execution uses daily closes only:** same-day close (optimistic) or next-day
  close. There are no opening or intraday prices, and no limit or stop orders.
- **Strategies are all-in / all-out** on a single ticker (SPY).
- **Small sample:** one market, three hand-picked calendar periods, and strategy
  parameters (50/200) that were not optimised or validated out of sample.
- **Data source:** `yfinance` is an unofficial, community-maintained library;
  Yahoo's data may contain errors or be revised.
- **Benchmark timing in single-run analytics:** `python -m simulator.analytics`
  always buys SPY at the first day's close, so a `next_close` run starts one day
  after its benchmark. (The strategy comparison does not have this difference.)
- Rejected and unfilled orders are reported when a strategy runs but are not saved
  to the database.

---

## Testing

```bash
pytest
```

**253 tests, all passing** (verified with pytest 9.1.1 on Python 3.14.8, macOS).

| Test file | Tests | Covers |
|---|---|---|
| `test_portfolio.py` | 32 | Cash, commission, fractional shares, average cost, validation |
| `test_execution.py` | 32 | Next-day execution: day *t* orders fill on *t+1*, sizing, rejections |
| `test_strategies.py` | 30 | Crossover signals, buy-and-hold, look-ahead checks, registry |
| `test_analytics.py` | 37 | Every metric against hand-calculated values, benchmark, edge cases |
| `test_comparison.py` | 25 | Date alignment, period boundaries, warm-up, look-ahead bias |
| `test_storage.py` | 20 | Save/load round trips, duplicates, transactions, error cases |
| `test_dashboard_data.py` | 18 | Dashboard tables and chart data |
| `test_engine.py` | 17 | Replay order, snapshots, no future prices |
| `test_data_loader.py` | 15 | CSV validation and trading-day alignment |
| `test_run_strategy.py` | 15 | Running and saving strategies, exact 50/200 crossing days, re-runs |
| `test_dashboard_app.py` | 9 | The Streamlit app, run headlessly with Streamlit's `AppTest` |
| `test_download.py` | 3 | Cleaning downloaded data (no network) |

Expected values in the tests are worked out by hand from small made-up price
series. Look-ahead protection is tested directly: changing prices after a given
day must not change any earlier decision.

---

## Project structure

```
simulator/
  config.py         tickers, benchmark, start date, file paths, risk-free rate
  download.py       download prices from Yahoo Finance (CLI)
  data_loader.py    load and validate price CSVs; PriceHistory
  portfolio.py      cash, positions, trades, commission (Decimal)
  engine.py         day-by-day Simulation and execution timing
  strategies.py     BuyAndHold, MovingAverageCrossover, strategy registry
  run_strategy.py   run and save registered strategies (CLI)
  storage.py        SQLite persistence (CLI lists saved runs)
  analytics.py      performance metrics and SPY benchmark (CLI report)
  comparison.py     fair multi-period strategy comparison (CLI report)
dashboard/
  app.py            Streamlit page layout and charts
  data.py           data preparation for the dashboard (no Streamlit)
examples/
  buy_and_hold.py   minimal example: buy 10 AAPL shares and hold
tests/              pytest suite (253 tests)
.streamlit/
  config.toml       dashboard theme; disables usage statistics
data/               downloaded prices and SQLite database (not in Git)
```
