"""Performance analytics: how well did a simulation do, and did it beat the S&P 500?

Every metric is calculated from a "value series": the starting cash, followed by
the portfolio's total value at the close of each trading day. Starting from the
cash (rather than the first day's close) means the first day's commissions count
against the result.

Look-ahead bias: these metrics only look back at results that have already
happened, and nothing here feeds back into trading decisions. The benchmark is
built with the same engine as any strategy, buying SPY at the first day's close
using only that day's price.

Run from the project root (after saving a run, e.g. python -m examples.buy_and_hold):
    python -m simulator.analytics               # the most recently saved run
    python -m simulator.analytics buy_and_hold  # a run by name
"""

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from simulator.config import BENCHMARK, DATABASE_PATH, PRICE_DATA_DIR, RISK_FREE_RATE
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import DailySnapshot, Simulation
from simulator.portfolio import Portfolio
from simulator.storage import StorageError, list_runs, load_run_settings, load_snapshots
from simulator.strategies import BuyAndHold

TRADING_DAYS_PER_YEAR = 252  # US stock markets are open about 252 days a year
DAYS_PER_YEAR = 365.25  # calendar days, allowing for leap years


@dataclass(frozen=True)
class Drawdown:
    """The biggest fall from a high point to a later low point."""

    fraction: float  # 0.25 means the value fell 25% below its earlier peak
    peak_day: date | None  # both None when the value never fell
    trough_day: date | None


@dataclass(frozen=True)
class Performance:
    """Every metric for one value series. Fractions: 0.10 means 10%."""

    first_day: date
    last_day: date
    starting_value: Decimal
    final_value: Decimal
    total_return: Decimal  # final_value - starting_value, in dollars
    total_return_pct: float
    annualised_return: float | None  # None when it cannot be calculated
    max_drawdown: Drawdown
    volatility: float | None  # yearly; None with fewer than 2 daily returns
    sharpe_ratio: float | None  # None with fewer than 2 returns or zero volatility
    daily_returns: list[float] = field(repr=False)

    @property
    def years(self) -> float:
        return (self.last_day - self.first_day).days / DAYS_PER_YEAR


@dataclass(frozen=True)
class Comparison:
    """A strategy and the SPY benchmark over the same days."""

    strategy: Performance
    benchmark: Performance

    @property
    def excess_return_pct(self) -> float:
        """Strategy total return minus benchmark total return. Positive = beat SPY."""
        return self.strategy.total_return_pct - self.benchmark.total_return_pct

    @property
    def excess_annualised_return(self) -> float | None:
        if self.strategy.annualised_return is None or self.benchmark.annualised_return is None:
            return None
        return self.strategy.annualised_return - self.benchmark.annualised_return


# ----- individual metrics ------------------------------------------------
# Each function does one calculation, so each can be tested on its own.

def total_return_pct(starting_value, final_value) -> float:
    """(final / start) - 1. Example: 10,000 -> 11,000 is 0.10 (+10%)."""
    if starting_value <= 0:
        raise ValueError("starting value must be positive")
    return float(final_value) / float(starting_value) - 1


def annualised_return(starting_value, final_value, first_day: date, last_day: date) -> float | None:
    """The steady yearly return that would turn starting_value into final_value.

    Formula: (final / start) ** (1 / years) - 1, with years = calendar days / 365.25.
    It is also called CAGR (compound annual growth rate).

    Limitations: it hides the ups and downs along the way, and for periods
    shorter than about a year it "stretches" a short result to a whole year,
    which can give huge, meaningless numbers. Returns None for a zero-length
    period or when the result is too large to represent.
    """
    days = (last_day - first_day).days
    if days <= 0:
        return None
    growth = total_return_pct(starting_value, final_value) + 1
    if growth <= 0:
        return -1.0  # everything was lost
    try:
        return growth ** (DAYS_PER_YEAR / days) - 1
    except OverflowError:
        return None


def daily_returns(values: list) -> list[float]:
    """Each day's change as a fraction of the previous day's value.

    Example: [100, 110, 99] -> [0.10, -0.10].
    """
    floats = [float(value) for value in values]
    if any(value < 0 for value in floats):
        raise ValueError("values cannot be negative")
    if any(value == 0 for value in floats[:-1]):
        raise ValueError("value reached zero before the last day; later returns are undefined")
    return [today / yesterday - 1 for yesterday, today in zip(floats, floats[1:])]


def max_drawdown(values: list, days: list[date]) -> Drawdown:
    """Largest fall from a running peak to a later value.

    Only values up to each day are used to find that day's peak.
    Example: [100, 120, 90, 130] -> 0.25 (from 120 down to 90).
    """
    if not values or len(values) != len(days):
        raise ValueError("need one day for each value, and at least one value")
    peak_value, peak_day = values[0], days[0]
    worst = Drawdown(0.0, None, None)
    for value, day in zip(values, days):
        if value > peak_value:
            peak_value, peak_day = value, day
        elif peak_value > 0:
            fall = 1 - float(value) / float(peak_value)
            if fall > worst.fraction:
                worst = Drawdown(fall, peak_day, day)
    return worst


def drawdown_series(values: list) -> list[float]:
    """Each value's distance below the highest value so far, as a fraction.

    0.0 means "at a new high"; -0.25 means "25% below the earlier peak".
    The most negative entry is the max drawdown (with the sign flipped).
    Example: [100, 120, 90, 130] -> [0.0, 0.0, -0.25, 0.0].
    """
    result = []
    peak = None
    for value in (float(v) for v in values):
        peak = value if peak is None else max(peak, value)
        result.append(value / peak - 1 if peak > 0 else 0.0)
    return result


def volatility(returns: list[float]) -> float | None:
    """Yearly volatility: the standard deviation of daily returns x sqrt(252).

    Uses the sample standard deviation, so it needs at least 2 returns.
    """
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: list[float], risk_free_rate: float = RISK_FREE_RATE) -> float | None:
    """Return earned per unit of risk, compared with a risk-free investment.

    Formula: mean(daily return - daily risk-free rate) / stdev(daily returns) x sqrt(252).
    The yearly risk_free_rate is turned into a daily one by compounding:
    (1 + rate) ** (1/252) - 1. Returns None when volatility is zero (for
    example a portfolio that only holds cash) or there are fewer than 2 returns.
    """
    if len(returns) < 2:
        return None
    daily_risk_free = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = [r - daily_risk_free for r in returns]
    spread = statistics.stdev(excess)
    if spread == 0:
        return None
    return statistics.mean(excess) / spread * math.sqrt(TRADING_DAYS_PER_YEAR)


# ----- whole-run analysis ------------------------------------------------

def analyse(
    snapshots: list[DailySnapshot], starting_cash, risk_free_rate: float = RISK_FREE_RATE
) -> Performance:
    """Calculate every metric for one run's daily snapshots."""
    if not snapshots:
        raise ValueError("no snapshots to analyse")
    starting_cash = Decimal(str(starting_cash))
    if starting_cash <= 0:
        raise ValueError("starting cash must be positive")
    snapshot_days = [snap.day for snap in snapshots]
    if any(later <= earlier for earlier, later in zip(snapshot_days, snapshot_days[1:])):
        raise ValueError("snapshots must be in date order, one per day")

    # The starting cash is the value at the first day's close, before any trades.
    days = [snapshot_days[0], *snapshot_days]
    values = [starting_cash, *(snap.total_value for snap in snapshots)]
    returns = daily_returns(values)
    final_value = values[-1]

    return Performance(
        first_day=days[0],
        last_day=days[-1],
        starting_value=starting_cash,
        final_value=final_value,
        total_return=final_value - starting_cash,
        total_return_pct=total_return_pct(starting_cash, final_value),
        annualised_return=annualised_return(starting_cash, final_value, days[0], days[-1]),
        max_drawdown=max_drawdown(values, days),
        volatility=volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        daily_returns=returns,
    )


def benchmark_snapshots(
    spy_prices: PriceHistory,
    days: list[date],
    starting_cash,
    commission,
    ticker: str = BENCHMARK,
) -> list[DailySnapshot]:
    """Buy-and-hold SPY over exactly the given days, using the normal engine.

    On the first day it spends all the cash it can (after one commission) on
    SPY at that day's close, then holds until the last day.
    """
    if not days:
        raise ValueError("no days to build a benchmark for")
    try:
        sim = Simulation(spy_prices, Portfolio(starting_cash, commission), start=days[0], end=days[-1])
    except ValueError:
        raise DataError(f"no {ticker} prices between {days[0]} and {days[-1]}")

    # The same strategy users can run themselves (see simulator.strategies).
    sim.run(BuyAndHold(ticker))

    by_day = {snap.day: snap for snap in sim.history}
    missing = [day for day in days if day not in by_day]
    if missing:
        raise DataError(
            f"{ticker} has no price on {len(missing)} of the strategy's days (first: {missing[0]})"
        )
    # Keep only the strategy's days so both are measured on the same dates.
    return [by_day[day] for day in days]


def compare(
    snapshots: list[DailySnapshot],
    spy_prices: PriceHistory,
    starting_cash,
    commission,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Comparison:
    """Analyse a strategy and SPY buy-and-hold over the same dates."""
    if not snapshots:
        raise ValueError("no snapshots to analyse")
    days = [snap.day for snap in snapshots]
    benchmark = benchmark_snapshots(spy_prices, days, starting_cash, commission)
    return Comparison(
        strategy=analyse(snapshots, starting_cash, risk_free_rate),
        benchmark=analyse(benchmark, starting_cash, risk_free_rate),
    )


def compare_saved_run(
    name: str,
    db_path: Path = DATABASE_PATH,
    data_dir: Path = PRICE_DATA_DIR,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Comparison:
    """Load a saved run from the database and compare it with SPY."""
    snapshots = load_snapshots(name, db_path)
    starting_cash, commission = load_run_settings(name, db_path)
    spy_prices = PriceHistory.from_directory(data_dir, [BENCHMARK])
    return compare(snapshots, spy_prices, starting_cash, commission, risk_free_rate)


# ----- printing a report -------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _money(value: Decimal) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def _drawdown(drawdown: Drawdown) -> str:
    return f"{-drawdown.fraction:.2%}" if drawdown.fraction else "0.00%"


def format_report(name: str, comparison: Comparison, risk_free_rate: float = RISK_FREE_RATE) -> str:
    s, b = comparison.strategy, comparison.benchmark
    rows = [
        ("Starting value", _money(s.starting_value), _money(b.starting_value)),
        ("Final value", _money(s.final_value), _money(b.final_value)),
        ("Total return", _money(s.total_return), _money(b.total_return)),
        ("Total return %", _pct(s.total_return_pct), _pct(b.total_return_pct)),
        ("Annualised return", _pct(s.annualised_return), _pct(b.annualised_return)),
        ("Max drawdown", _drawdown(s.max_drawdown), _drawdown(b.max_drawdown)),
        ("Volatility (yearly)", _pct(s.volatility).lstrip("+"), _pct(b.volatility).lstrip("+")),
        ("Sharpe ratio", _ratio(s.sharpe_ratio), _ratio(b.sharpe_ratio)),
    ]
    lines = [
        f"Run {name!r} vs {BENCHMARK} buy-and-hold",
        f"{s.first_day} to {s.last_day} ({s.years:.1f} years, {len(s.daily_returns)} trading days)",
        "",
        f"{'':22}{'Strategy':>14}{BENCHMARK:>14}",
        *(f"{label:22}{left:>14}{right:>14}" for label, left, right in rows),
        "",
    ]
    for label, drawdown in (("Strategy", s.max_drawdown), (BENCHMARK, b.max_drawdown)):
        if drawdown.peak_day:
            lines.append(f"{label} max drawdown: peak {drawdown.peak_day}, low {drawdown.trough_day}")

    excess = comparison.excess_return_pct
    verdict = "beat" if excess > 0 else "trailed" if excess < 0 else "matched"
    lines += [
        "",
        f"The strategy {verdict} {BENCHMARK} by {abs(excess) * 100:.2f} percentage points of total return.",
        f"Sharpe ratio assumes a risk-free rate of {risk_free_rate:.2%} a year.",
    ]
    if s.years < 1:
        lines.append("Note: with less than a year of data, annualised figures can be misleading.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a saved run with the S&P 500.")
    parser.add_argument("run", nargs="?", help="saved run name (default: the latest saved run)")
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=RISK_FREE_RATE,
        help=f"yearly risk-free rate for the Sharpe ratio, e.g. 0.04 for 4%% (default {RISK_FREE_RATE})",
    )
    args = parser.parse_args(argv)

    try:
        name = args.run
        if name is None:
            runs = list_runs()
            if not runs:
                print("No saved runs yet. Try: python -m examples.buy_and_hold")
                return 1
            name = runs[-1]
        comparison = compare_saved_run(name, risk_free_rate=args.risk_free_rate)
    except (StorageError, DataError, ValueError) as error:
        print(f"Could not analyse: {error}")
        return 1

    print(format_report(name, comparison, args.risk_free_rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
