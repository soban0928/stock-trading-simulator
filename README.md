# Stock Trading Simulator

A day-by-day stock trading simulator using historical US market data. You start
with $10,000 of virtual cash, trade US stocks at each day's closing price, and
compare your performance with the S&P 500.

> Work in progress. This README will be completed as the project is built.

## Current status

- [x] Stage 1: core portfolio engine (cash, buy/sell, commission, fractional shares)
- [x] Stage 2: historical price data loading
- [x] Stage 3: day-by-day replay engine
- [x] Stage 4: SQLite storage
- [x] Stage 5: performance analytics vs S&P 500
- [x] Stage 6: Streamlit dashboard
- [x] Stage 7: trading strategies (SPY buy-and-hold, 50/200-day moving-average crossover)
- [x] Next-trading-day execution (`--execution next_close`)
- [x] Stage 8: fair strategy comparison and robustness periods

## Setup (macOS)

```bash
cd "path/to/Stock Trading Simulator"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Getting price data

Prices are downloaded from Yahoo Finance with the unofficial
[yfinance](https://github.com/ranaroussi/yfinance) library and saved as CSV
files in `data/prices/`. They are not included in this repository; download
them yourself:

```bash
python -m simulator.download
```

The stock list, benchmark (SPY) and start date are set in `simulator/config.py`.
Trades use the daily close adjusted for stock splits but not dividends.

## Running a simulation

`simulator/engine.py` replays the price data one trading day at a time. On each
day you can only see prices up to that day's close, and every trade happens at
that close (with the commission added), so no decision can use future prices.

```python
from simulator.config import PRICE_DATA_DIR, TICKERS
from simulator.data_loader import PriceHistory
from simulator.engine import Simulation

prices = PriceHistory.from_directory(PRICE_DATA_DIR, TICKERS)
sim = Simulation(prices)  # $10,000 cash, $1.00 commission per trade

def buy_and_hold(sim):
    if not sim.trades:
        sim.buy("AAPL", 10)

history = sim.run(buy_and_hold)
print(history[-1].day, history[-1].total_value, history[-1].profit_loss)
```

`sim.history` holds one `DailySnapshot` per day (cash, market value, total
value, profit/loss, holdings) and `sim.trades` holds every executed trade.

A ready-made version of this is in `examples/buy_and_hold.py`:

```bash
python -m examples.buy_and_hold
```

## Trading strategies

`simulator/strategies.py` holds ready-made strategies, each registered under a
unique name. Run them with:

```bash
python -m simulator.run_strategy --list                     # what's available
python -m simulator.run_strategy spy_buy_and_hold           # run one
python -m simulator.run_strategy spy_sma_crossover_50_200
python -m simulator.run_strategy                            # run all of them
python -m simulator.run_strategy --execution next_close     # fill orders a day later
python -m simulator.analytics spy_sma_crossover_50_200      # compare with SPY
```

Each run is saved to the database under the strategy's name, with $10,000
starting cash and a $1.00 commission per trade. Running a strategy again
**replaces** its saved copy (like the example script), so there is always one
copy per strategy and the dashboard picks it up straight away. Strategies are
deterministic: the same price data always gives the same trades.

| Name | What it does |
|---|---|
| `spy_buy_and_hold` | Spends all the cash on SPY at the first day's close (after one commission), then holds |
| `spy_sma_crossover_50_200` | Buys SPY with all the cash when its 50-day average crosses **above** its 200-day average; sells every share when it crosses **below** |

### How the moving-average crossover decides

- The **50-day simple moving average** is the mean of the last 50 closes,
  including today's; the 200-day one uses the last 200.
- A **crossing** compares today with the previous trading day: "crosses above"
  means the 50-day was at or below the 200-day yesterday and is above it today.
- **No trades until both averages exist for today and yesterday** (201 trading
  days of prices, so the first possible signal is in October 2015).
- A buy signal while already invested, or a sell signal with nothing to sell,
  does nothing. If the 50-day is already above the 200-day when the averages
  first exist, that is not a crossing, so the strategy waits in cash for the
  next upward cross.

### Assumptions and limitations

- **Execution timing.** By default a signal and its trade both use the same
  day's close, which isn't possible in real life. Run with
  `--execution next_close` to fill each order at the next trading day's close
  instead; see [Execution timing](#execution-timing) below.
- **All in or all out.** Every buy spends all available cash (shares rounded
  down to 6 decimal places, so it never overspends); every sell sells
  everything. Cash earns no interest while out of the market.
- **No dividends, taxes, slippage or bid/ask spread**, only the $1 commission.
  Price data is split-adjusted but not dividend-adjusted, which especially
  understates the buy-and-hold strategies.
- **Crossovers react late.** They are built from averages of past prices, so
  they often buy after a rally has started and sell after a fall has happened.
  In this data the sell signal for the 2020 crash came on 30 March 2020, a week
  after the bottom, so the strategy took the full crash drawdown.
- **One price history, one result.** A strategy that did well (or badly) on
  2015–2026 SPY data says little about how it would do in the future.

`spy_buy_and_hold` uses the same `BuyAndHold` strategy that analytics uses as
the S&P 500 benchmark, so its results match the benchmark exactly.

### Execution timing

A strategy runs after day *t*'s close is known, because it uses that close to
make its decision. The question is when its order fills. The engine has two
settings (`Simulation(prices, execution=...)`):

| Setting | Order placed on day *t* fills at | Use |
|---|---|---|
| `same_close` (default) | day *t*'s close, immediately | Simple, and how all earlier runs were made. Optimistic: in reality you can't see a close and still trade at it |
| `next_close` | day *t+1*'s close | Realistic timing: the decision only uses information you'd have had, and the trade happens afterwards |

How `next_close` works:

- `sim.submit_order(ticker, "BUY" or "SELL", quantity)` queues the order. It
  fills when the simulation moves to the next trading day, before the strategy
  runs for that day. `buy_all` and `sell_all` (used by both strategies) do this
  automatically. Calling `sim.buy()` or `sim.sell()` directly raises an error,
  so nothing can fill on the signal day by mistake.
- **"All cash" and "all shares" orders are sized on the fill day**, using that
  day's cash, holdings and price, with one commission and shares rounded down to
  6 decimal places. A price jump overnight can't make the portfolio overspend.
- A **fixed-quantity** order that can't be afforded on the fill day is recorded
  in `sim.rejected_orders` with the reason; no commission is charged and the
  simulation carries on.
- On a fill day, **sells fill before buys**, so a sale's cash can fund a purchase.
- An order placed on the **last day** has no next day and is never filled; it
  appears in `sim.unfilled_orders`.
- Runs are saved with a `__next_close` suffix (for example
  `spy_sma_crossover_50_200__next_close`), so they sit beside the same-day
  versions in the database, analytics and dashboard. Re-running replaces only
  that version's copy.

Limitations:

- **No opening prices.** The price files only contain `date,close`, so the
  engine can't fill at the next day's *open*, which is when an order sent
  overnight would really fill. Supporting that would need: `download.py` to also
  keep yfinance's `Open` column (split-adjusted, like the close, with
  `auto_adjust=False`), CSVs with a `date,open,close` layout, `data_loader.py` to
  validate the opens like the closes, a `next_open` setting in the engine, and a
  fresh download of every file. Opening prices are never estimated.
- The S&P 500 benchmark in analytics still buys at the first day's close, so a
  `next_close` buy-and-hold run starts one day later than its benchmark.
- Rejected and unfilled orders are reported when a strategy runs, but are not
  saved to the database.
- A one-day delay isn't automatically worse; it changes the fill prices. On this
  data, `next_close` happened to do slightly better for both strategies. Its
  value is that it can't benefit from knowing a price before trading at it.

To add your own strategy, write a class or function that takes the
`Simulation` (like `MovingAverageCrossover`) and add a `StrategySpec` with a new
unique name to `STRATEGIES`. Duplicate names are rejected.

## Fair strategy comparison and robustness periods

`simulator/comparison.py` runs every registered strategy side by side, over the
full history and over separate periods, to see whether a result holds up in
different market conditions or depends on one lucky stretch.

```bash
python -m simulator.comparison                                  # same-day close
python -m simulator.comparison --execution next_close --risk-free-rate 0.04
```

The same comparison is in the dashboard: choose **Strategy comparison** in the
sidebar. It runs in memory and saves nothing, so saved runs are not changed.

### Methodology

| Period | Dates used (this data) | Earlier trading days available for warm-up |
|---|---|---|
| Full history | 2015-01-02 → 2026-10-21 | 0 |
| 2015–2019 | 2015-01-02 → 2019-12-31 | 0 |
| 2020–2022 | 2020-01-02 → 2022-12-30 | 1,258 |
| 2023–2026 | 2023-01-03 → 2026-10-21 (data ends here) | 2,014 |

Within each period, every strategy gets **identical conditions**:

- the same trading days (the first and last trading day inside the period);
- $10,000 starting cash and a $1.00 commission per trade;
- the same execution setting (`same_close` or `next_close`);
- a fresh start: in cash on the first day, with nothing carried over from an
  earlier period.

**The benchmark is the `spy_buy_and_hold` strategy itself**, run under those
same conditions. Under `next_close` it buys one day late, like every other
strategy, so its start date is consistent with theirs. (The analytics report
for a single saved run still buys SPY at the first day's close; see
[Execution timing](#execution-timing).)

Each period reports total return, annualised return, max drawdown, volatility
and Sharpe ratio (calculated by `simulator/analytics.py`, as defined below),
plus the number of trades and the share of days the strategy ended invested.

### Indicator warm-up and look-ahead bias

- Moving averages use closes **up to and including the current day**. On the
  first day of a period, that includes prices from *before* the period: they
  are already in the past, so using them is legitimate. This is the warm-up.
- **Nothing after the current day is used, and nothing after the period's end
  is used at all.** Tests check this two ways: cutting the price data off at the
  period's end gives identical results, and changing every later price changes
  nothing.
- **2015–2019 (and Full history) have no earlier data**, because the price files
  start on 2015-01-02. The 50/200 crossover needs 201 closes for its first
  possible signal, which comes on **2015-10-19**. Until then it can only hold cash,
  while buy-and-hold is invested from day 1. This difference can't be avoided
  with this data, and it lowers the crossover's 2015–2019 result.
- Each period's report notes how the crossover's averages stood on its first day.

### Assumptions and limitations

- **Fresh starts change results.** A crossover only buys on an upward *cross*. On
  2020-01-02 the 50-day average was already above the 200-day average, so the
  fresh 2020–2022 run waited in cash until the next upward cross (2020-07-09)
  and so missed the 2020 crash. The continuous full-history run was invested
  through that crash. The crossover's strong 2020–2022 numbers are partly an
  effect of where the period starts, not evidence of skill.
- **Few, overlapping, hand-picked periods.** Three periods of one market (SPY) are
  a small sample. The periods were chosen by calendar year, not to favour any
  strategy, but results can change a lot if the boundaries move.
- **The latest period is partial** (it ends with the data on 2026-10-21), and
  annualised figures for shorter periods are less reliable.
- **Strategy settings were not tuned per period**, and the 50/200 windows are a
  common convention, not an optimised choice. Optimising them on this data and
  then judging them on the same data would overstate results.
- The same data limits as elsewhere apply: no dividends, no interest on cash, no
  slippage or taxes, and either same-day or next-day *close* execution only.
- **None of these results show that a strategy is profitable or robust in the
  future.** They describe what these rules would have done on this history.

## Saving results (SQLite)

`simulator/storage.py` saves a simulation's trades and daily snapshots to
`data/simulator.db`, a single SQLite file. Like the price data, it is
generated locally and kept out of Git by `.gitignore`.

```python
from simulator.storage import list_runs, load_snapshots, load_trades, save_simulation

save_simulation(sim, "my_strategy")       # after sim.run(...)
print(list_runs())                        # ['my_strategy']
trades = load_trades("my_strategy")       # list of Trade
history = load_snapshots("my_strategy")   # list of DailySnapshot
```

Every saved run needs a unique name. Saving the same name again raises
`RunExistsError`, so nothing is duplicated by accident; pass `overwrite=True`
to replace the old copy instead. The database file and its tables are created
automatically the first time you save.

To see what is saved:

```bash
python -m simulator.storage
```

The tables are `runs`, `trades`, `snapshots` and `holdings`. Money and share
quantities are stored as text so they stay exact. You can also open the file
with the `sqlite3` tool that comes with macOS:

```bash
sqlite3 data/simulator.db "SELECT trade_date, ticker, side, quantity, price FROM trades;"
```

## Performance analytics

`simulator/analytics.py` measures a saved run and compares it with simply
buying the S&P 500 (SPY) over the same dates.

```bash
python -m examples.buy_and_hold          # save a run first
python -m simulator.analytics            # analyse the latest saved run
python -m simulator.analytics buy_and_hold --risk-free-rate 0.04
```

You can also use it from Python, on a saved run or a simulation in memory:

```python
from simulator.analytics import analyse, compare_saved_run

comparison = compare_saved_run("buy_and_hold")
print(comparison.strategy.annualised_return, comparison.benchmark.annualised_return)

performance = analyse(sim.history, sim.portfolio.starting_cash)
```

### How each metric is calculated

All metrics use the **value series**: the starting cash, followed by the
portfolio's total value at each day's close. Percentages are stored as
fractions (0.10 = 10%).

| Metric | Meaning | Formula |
|---|---|---|
| Total return | Dollars gained or lost | final value − starting cash |
| Total return % | Gain as a share of what you started with | final ÷ start − 1 |
| Annualised return | Steady yearly return that gives the same result (CAGR) | (final ÷ start) ^ (1 ÷ years) − 1, where years = calendar days ÷ 365.25 |
| Daily returns | Each day's change | today ÷ yesterday − 1 |
| Volatility | How much daily returns jump around, per year | sample standard deviation of daily returns × √252 |
| Max drawdown | Biggest fall from a high point to a later low | largest (1 − value ÷ highest value so far) |
| Sharpe ratio | Return per unit of risk, above a risk-free investment | mean(daily return − daily risk-free) ÷ std. dev. of daily returns × √252 |

252 is the usual number of US trading days in a year.

**Limitations and assumptions**

- **Annualised return** hides the ups and downs, and for periods shorter than
  about a year it stretches a short result to a full year, which can give huge,
  meaningless numbers. The report warns when there is less than a year of data.
- **Sharpe ratio** assumes a risk-free rate of **0%** by default
  (`RISK_FREE_RATE` in `simulator/config.py`). Real Treasury-bill rates are
  usually higher, which lowers the Sharpe ratio; try `--risk-free-rate 0.04`.
- **Undefined values** are shown as `n/a` instead of a misleading number:
  annualised return over zero days, volatility or Sharpe ratio with fewer than
  two daily returns, and Sharpe ratio when volatility is zero (an all-cash
  portfolio).
- The first daily return compares the first day's close with the starting
  cash, so it includes that day's commissions.

### The S&P 500 benchmark

The benchmark runs through the same engine as any strategy: on the first day it
spends all the starting cash (minus one commission) on SPY at that day's close,
rounding the share count down so it never overspends, and holds until the end.
It is measured on exactly the strategy's dates; if SPY has no price on one of
them, you get an error instead of a mismatched comparison. Like the strategy,
SPY prices are adjusted for splits but not dividends, so both returns leave out
dividends.

**No look-ahead bias:** the benchmark's purchase uses only the first day's
price, max drawdown compares each day only with earlier highs, and analytics
are calculated after the simulation ends, so they never influence a trade.

## Dashboard

An interactive Streamlit dashboard for browsing saved runs. It only reads the
database: it never places trades, and no real money is involved.

**Install** (once; Streamlit is listed in `requirements.txt`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Launch** from the project folder:

```bash
python -m examples.buy_and_hold   # save at least one run first
streamlit run dashboard/app.py
```

Your browser opens at http://localhost:8501. Press `Ctrl+C` in the terminal to
stop the dashboard.

What it shows:

- **Overview:** starting cash, final value, total profit/loss and total return,
  with the difference from the S&P 500 in percentage points (pp).
- **Equity curve:** portfolio value each day next to SPY buy-and-hold. Hover
  for exact values on any date.
- **Risk and return:** annualised return, max drawdown, volatility and Sharpe
  ratio, plus a side-by-side table against SPY.
- **Drawdown chart:** how far each day's value sits below its previous peak.
- **Records:** the trade history and daily portfolio history (downloadable as CSV).
- **Sidebar:** choose a saved run, and set the risk-free rate for the Sharpe ratio.
- **Strategy comparison** (sidebar view switch): every strategy against SPY over
  each robustness period, with same-day or next-day execution. Works even if
  nothing is saved yet; it only needs the price files.

If nothing is saved yet, the page explains how to save a run. If the SPY price
file is missing, it shows the strategy alone with a warning.

How it is organised:

| File | Role |
|---|---|
| `dashboard/data.py` | Loads a run and builds tables and chart data (no Streamlit; tested) |
| `dashboard/app.py` | Page layout, charts and messages only |
| `.streamlit/config.toml` | Dark theme; turns off Streamlit's usage statistics |

All numbers come from `simulator/storage.py` and `simulator/analytics.py`; the
dashboard does not recalculate them.
