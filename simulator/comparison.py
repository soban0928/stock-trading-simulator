"""Fair side-by-side comparison of strategies, across several historical periods.

"Fair" here means every strategy in a period gets exactly the same:
  - trading days (same first and last day, same calendar),
  - starting cash and commission,
  - execution setting (same_close or next_close).
The S&P 500 benchmark is the spy_buy_and_hold strategy itself, run with those
same settings, so it starts on the same day as every other strategy.

Each period is a fresh simulation that starts in cash on the period's first
trading day. Strategies may *look back* at prices from before the period
(that's how a 200-day average is "warmed up"), but they never see a price
after the current day, and nothing after the period's end is used at all.

Nothing here is saved to the database; saved runs are left untouched.

Run from the project root:
    python -m simulator.comparison
    python -m simulator.comparison --execution next_close --risk-free-rate 0.04
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from simulator.analytics import Performance, analyse
from simulator.config import BENCHMARK, PRICE_DATA_DIR, RISK_FREE_RATE
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import EXECUTION_MODES, SAME_CLOSE, DailySnapshot, Simulation
from simulator.portfolio import Portfolio
from simulator.strategies import STRATEGIES, MovingAverageCrossover, get_strategy

BENCHMARK_STRATEGY = "spy_buy_and_hold"


@dataclass(frozen=True)
class Period:
    """A date range to test over. None means "from the start / to the end of the data"."""

    name: str
    start: date | None = None
    end: date | None = None


PERIODS = (
    Period("Full history"),
    Period("2015–2019", date(2015, 1, 1), date(2019, 12, 31)),
    Period("2020–2022", date(2020, 1, 1), date(2022, 12, 31)),
    Period("2023–2026", date(2023, 1, 1), date(2026, 12, 31)),
)


@dataclass(frozen=True)
class StrategyResult:
    """One strategy's results over one period."""

    strategy: str
    performance: Performance
    trades: int
    invested_fraction: float  # share of days that ended holding any shares
    start_note: str | None  # how the strategy's indicators stood on the first day
    snapshots: list[DailySnapshot] = field(repr=False)


@dataclass(frozen=True)
class PeriodResult:
    """Every strategy's results over one period, on identical trading days."""

    period: Period
    first_day: date
    last_day: date
    trading_days: int
    history_days_before: int  # trading days available for warming up indicators
    results: list[StrategyResult]

    def result(self, strategy: str) -> StrategyResult:
        return next(r for r in self.results if r.strategy == strategy)


def invested_fraction(snapshots: list[DailySnapshot]) -> float:
    """Share of days on which the portfolio ended the day holding any shares."""
    if not snapshots:
        return 0.0
    return sum(1 for snap in snapshots if snap.holdings) / len(snapshots)


def start_note(strategy, prices: PriceHistory, first_day: date) -> str | None:
    """Explain how a crossover strategy's averages stood on a period's first day.

    Uses only closes up to and including first_day. Every period starts in
    cash, and a crossover only buys on an upward *cross*, so this state can
    decide whether the strategy spends the first months of a period in cash.
    """
    if not isinstance(strategy, MovingAverageCrossover):
        return None
    closes = prices.closes_through(strategy.ticker, first_day)
    short, long = strategy.short_window, strategy.long_window
    if len(closes) < long + 1:
        # The first signal needs long_window + 1 closes (today's and yesterday's averages).
        days = prices.trading_days
        ready_index = days.index(first_day) + (long + 1 - len(closes))
        ready = days[ready_index] if ready_index < len(days) else None
        return (f"warming up: not enough history for the {long}-day average; "
                f"first possible signal {ready or 'after the data ends'}")
    short_avg, long_avg = closes.iloc[-short:].mean(), closes.iloc[-long:].mean()
    if short_avg > long_avg:
        return (f"{short}-day average already above {long}-day at the start, so it waits "
                "in cash for the next upward cross")
    return f"{short}-day average below {long}-day at the start; buys on the next upward cross"


def run_period(
    prices: PriceHistory,
    period: Period,
    strategy_names: list[str],
    execution: str = SAME_CLOSE,
    starting_cash=None,
    commission=None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PeriodResult | None:
    """Run every strategy over one period. None if the data has no days in it."""
    days = [
        day for day in prices.trading_days
        if (period.start is None or day >= period.start) and (period.end is None or day <= period.end)
    ]
    if not days:
        return None

    defaults = Portfolio()
    cash = defaults.starting_cash if starting_cash is None else starting_cash
    fee = defaults.commission if commission is None else commission

    results = []
    for name in strategy_names:
        spec = get_strategy(name)
        sim = Simulation(prices, Portfolio(cash, fee), start=days[0], end=days[-1], execution=execution)
        sim.run(spec.strategy)
        results.append(
            StrategyResult(
                strategy=name,
                performance=analyse(sim.history, sim.portfolio.starting_cash, risk_free_rate),
                trades=len(sim.trades),
                invested_fraction=invested_fraction(sim.history),
                start_note=start_note(spec.strategy, prices, days[0]),
                snapshots=sim.history,
            )
        )

    return PeriodResult(
        period=period,
        first_day=days[0],
        last_day=days[-1],
        trading_days=len(days),
        history_days_before=sum(1 for day in prices.trading_days if day < days[0]),
        results=results,
    )


def compare_strategies(
    prices: PriceHistory,
    strategy_names: list[str] | None = None,
    periods=PERIODS,
    execution: str = SAME_CLOSE,
    starting_cash=None,
    commission=None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> list[PeriodResult]:
    """Run each strategy over each period. Periods with no data are left out.

    The benchmark (spy_buy_and_hold) is always included, listed first.
    """
    names = list(strategy_names or STRATEGIES)
    if BENCHMARK_STRATEGY in names:
        names.remove(BENCHMARK_STRATEGY)
    names.insert(0, BENCHMARK_STRATEGY)

    results = []
    for period in periods:
        result = run_period(prices, period, names, execution, starting_cash, commission, risk_free_rate)
        if result is not None:
            results.append(result)
    return results


def load_prices(data_dir: Path = PRICE_DATA_DIR, strategy_names: list[str] | None = None) -> PriceHistory:
    """Prices for every ticker the strategies (and the benchmark) need."""
    tickers = {BENCHMARK}
    for name in strategy_names or STRATEGIES:
        tickers.update(get_strategy(name).tickers)
    return PriceHistory.from_directory(data_dir, sorted(tickers))


def results_table(period_results: list[PeriodResult]) -> pd.DataFrame:
    """One row per (period, strategy), with raw numbers (fractions, not %)."""
    rows = []
    for pr in period_results:
        for r in pr.results:
            p = r.performance
            rows.append({
                "Period": pr.period.name,
                "From": pr.first_day,
                "To": pr.last_day,
                "Strategy": r.strategy,
                "Total return": p.total_return_pct,
                "Annualised return": p.annualised_return,
                "Max drawdown": -p.max_drawdown.fraction,
                "Volatility": p.volatility,
                "Sharpe ratio": p.sharpe_ratio,
                "Trades": r.trades,
                "Days invested": r.invested_fraction,
                "Final value": float(p.final_value),
                "Start note": r.start_note or "",
            })
    return pd.DataFrame(rows)


# ----- printing a report -------------------------------------------------

def _pct(value, signed=True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def format_report(period_results: list[PeriodResult], execution: str, risk_free_rate: float) -> str:
    lines = [
        f"Strategy comparison: same days, $10,000 start, $1 commission, execution={execution}, "
        f"risk-free rate {risk_free_rate:.2%}",
    ]
    header = f"  {'Strategy':28}{'Total':>10}{'Annual':>9}{'Max DD':>9}{'Vol':>8}{'Sharpe':>8}{'Trades':>8}{'Invested':>10}"
    for pr in period_results:
        lines += [
            "",
            f"{pr.period.name}: {pr.first_day} to {pr.last_day} ({pr.trading_days} trading days, "
            f"{pr.history_days_before} earlier days available for warm-up)",
            header,
        ]
        for r in pr.results:
            p = r.performance
            sharpe = "n/a" if p.sharpe_ratio is None else f"{p.sharpe_ratio:.2f}"
            lines.append(
                f"  {r.strategy:28}{_pct(p.total_return_pct):>10}{_pct(p.annualised_return):>9}"
                f"{_pct(-p.max_drawdown.fraction):>9}{_pct(p.volatility, signed=False):>8}"
                f"{sharpe:>8}{r.trades:>8}{r.invested_fraction:>10.0%}"
            )
        for r in pr.results:
            if r.start_note:
                lines.append(f"  * {r.strategy}: {r.start_note}")
    lines += [
        "",
        "Each period starts fresh in cash, so results can differ from the same dates inside a",
        "longer run. Past results on one market's history do not show that a strategy is",
        "profitable or robust.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare strategies fairly across periods.")
    parser.add_argument("--execution", choices=EXECUTION_MODES, default=SAME_CLOSE)
    parser.add_argument("--risk-free-rate", type=float, default=RISK_FREE_RATE)
    args = parser.parse_args(argv)
    try:
        prices = load_prices()
    except DataError as error:
        print(f"Could not load prices: {error}")
        return 1
    results = compare_strategies(prices, execution=args.execution, risk_free_rate=args.risk_free_rate)
    print(format_report(results, args.execution, args.risk_free_rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
