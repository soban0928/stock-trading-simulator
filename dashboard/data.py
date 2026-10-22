"""Turn a saved run into the tables and chart data the dashboard shows.

Nothing in here draws anything or imports Streamlit, so it can be tested with
pytest like the rest of the project. Every number comes from the existing
storage and analytics modules; this file only reshapes and formats them.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from simulator.analytics import (
    Comparison,
    Performance,
    analyse,
    benchmark_snapshots,
    drawdown_series,
)
from simulator.config import BENCHMARK, DATABASE_PATH, PRICE_DATA_DIR, RISK_FREE_RATE
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import DailySnapshot
from simulator.portfolio import Trade
from simulator.comparison import BENCHMARK_STRATEGY, PeriodResult, compare_strategies, load_prices
from simulator.run_strategy import RUN_NAME_SEPARATOR
from simulator.storage import load_run_settings, load_snapshots, load_trades
from simulator.strategies import STRATEGIES

STRATEGY_LABEL = "Strategy"
BENCHMARK_LABEL = f"S&P 500 ({BENCHMARK})"


@dataclass(frozen=True)
class RunData:
    """Everything the dashboard needs about one saved run."""

    name: str
    starting_cash: Decimal
    snapshots: list[DailySnapshot]
    trades: list[Trade]
    strategy: Performance
    benchmark_snapshots: list[DailySnapshot] | None  # None if SPY data is missing
    benchmark: Performance | None
    benchmark_error: str | None  # why the benchmark is missing, if it is

    @property
    def comparison(self) -> Comparison | None:
        if self.benchmark is None:
            return None
        return Comparison(self.strategy, self.benchmark)


EXECUTION_NOTES = {
    "": "Orders fill at the signal day's close.",
    "next_close": "Orders fill at the next trading day's close.",
}


def strategy_description(name: str) -> str | None:
    """What a run's strategy does and when its orders fill, for registered strategies.

    Run names look like 'spy_buy_and_hold' or 'spy_buy_and_hold__next_close'.
    """
    strategy_name, _, execution = name.partition(RUN_NAME_SEPARATOR)
    spec = STRATEGIES.get(strategy_name)
    if spec is None or execution not in EXECUTION_NOTES:
        return None
    return f"{spec.description} {EXECUTION_NOTES[execution]}"


def load_run(
    name: str,
    db_path: Path = DATABASE_PATH,
    data_dir: Path = PRICE_DATA_DIR,
    risk_free_rate: float = RISK_FREE_RATE,
) -> RunData:
    """Load a saved run and analyse it.

    Missing SPY data does not stop the strategy being shown: the benchmark is
    left out and benchmark_error says why. Database errors are raised as usual.
    """
    snapshots = load_snapshots(name, db_path)
    trades = load_trades(name, db_path)
    starting_cash, commission = load_run_settings(name, db_path)
    strategy = analyse(snapshots, starting_cash, risk_free_rate)

    bench_snapshots, benchmark, benchmark_error = None, None, None
    try:
        spy_prices = PriceHistory.from_directory(data_dir, [BENCHMARK])
        days = [snap.day for snap in snapshots]
        bench_snapshots = benchmark_snapshots(spy_prices, days, starting_cash, commission)
        benchmark = analyse(bench_snapshots, starting_cash, risk_free_rate)
    except DataError as error:
        bench_snapshots, benchmark_error = None, str(error)

    return RunData(
        name=name,
        starting_cash=starting_cash,
        snapshots=snapshots,
        trades=trades,
        strategy=strategy,
        benchmark_snapshots=bench_snapshots,
        benchmark=benchmark,
        benchmark_error=benchmark_error,
    )


# ----- chart data --------------------------------------------------------
# Charts want "long" tables: one row per (date, series) pair.

def _series(run: RunData) -> list[tuple[str, str, list[DailySnapshot]]]:
    series = [("strategy", STRATEGY_LABEL, run.snapshots)]
    if run.benchmark_snapshots is not None:
        series.append(("benchmark", BENCHMARK_LABEL, run.benchmark_snapshots))
    return series


def equity_curve(run: RunData) -> pd.DataFrame:
    """Total value at each day's close, for the strategy and the benchmark."""
    rows = [
        {"date": pd.Timestamp(snap.day), "key": key, "series": label, "value": float(snap.total_value)}
        for key, label, snapshots in _series(run)
        for snap in snapshots
    ]
    return pd.DataFrame(rows, columns=["date", "key", "series", "value"])


def drawdown_curve(run: RunData) -> pd.DataFrame:
    """How far each day's value is below its earlier peak (0 or negative)."""
    rows = []
    for key, label, snapshots in _series(run):
        # Start from the cash, as the analytics do, then drop that extra point.
        values = [run.starting_cash, *(snap.total_value for snap in snapshots)]
        for snap, drawdown in zip(snapshots, drawdown_series(values)[1:]):
            rows.append({"date": pd.Timestamp(snap.day), "key": key, "series": label, "drawdown": drawdown})
    return pd.DataFrame(rows, columns=["date", "key", "series", "drawdown"])


# ----- tables ------------------------------------------------------------

def trades_table(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": trade.trade_date,
                "Ticker": trade.ticker,
                "Action": trade.side,
                "Quantity": float(trade.quantity),
                "Price": float(trade.price),
                "Commission": float(trade.commission),
            }
            for trade in trades
        ],
        columns=["Date", "Ticker", "Action", "Quantity", "Price", "Commission"],
    )


def history_table(snapshots: list[DailySnapshot]) -> pd.DataFrame:
    """One row per day, newest first, so the latest day is at the top."""
    table = pd.DataFrame(
        [
            {
                "Date": snap.day,
                "Cash": float(snap.cash),
                "Market value": float(snap.market_value),
                "Total value": float(snap.total_value),
                "Profit/loss": float(snap.profit_loss),
            }
            for snap in snapshots
        ],
        columns=["Date", "Cash", "Market value", "Total value", "Profit/loss"],
    )
    return table.iloc[::-1].reset_index(drop=True)


def metrics_table(run: RunData) -> pd.DataFrame:
    """Side-by-side risk and return metrics, formatted for display."""
    columns = [(STRATEGY_LABEL, run.strategy), (BENCHMARK_LABEL, run.benchmark)]

    def row(label, get):
        return {"Metric": label, **{name: get(perf) if perf else "n/a" for name, perf in columns}}

    return pd.DataFrame(
        [
            row("Total return", lambda p: format_pct(p.total_return_pct)),
            row("Annualised return", lambda p: format_pct(p.annualised_return)),
            row("Max drawdown", lambda p: format_pct(-p.max_drawdown.fraction)),
            row("Volatility (annualised)", lambda p: format_pct(p.volatility, signed=False)),
            row("Sharpe ratio", lambda p: format_ratio(p.sharpe_ratio)),
        ]
    )


# ----- strategy comparison -----------------------------------------------
# All numbers come from simulator.comparison, which uses the analytics module.

STRATEGY_LABELS = {
    BENCHMARK_STRATEGY: f"{BENCHMARK} buy-and-hold (benchmark)",
    "spy_sma_crossover_50_200": "SMA 50/200 crossover",
}


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def load_comparison(
    data_dir: Path = PRICE_DATA_DIR, execution: str = "same_close", risk_free_rate: float = RISK_FREE_RATE
) -> list[PeriodResult]:
    """Run every registered strategy over every robustness period (nothing is saved)."""
    return compare_strategies(load_prices(data_dir), execution=execution, risk_free_rate=risk_free_rate)


def comparison_period_table(period_result: PeriodResult) -> pd.DataFrame:
    """One row per strategy for one period, formatted for display."""
    return pd.DataFrame(
        [
            {
                "Strategy": strategy_label(r.strategy),
                "Total return": format_pct(r.performance.total_return_pct),
                "Annualised return": format_pct(r.performance.annualised_return),
                "Max drawdown": format_pct(-r.performance.max_drawdown.fraction),
                "Volatility": format_pct(r.performance.volatility, signed=False),
                "Sharpe ratio": format_ratio(r.performance.sharpe_ratio),
                "Trades": r.trades,
                "Days invested": f"{r.invested_fraction:.0%}",
            }
            for r in period_result.results
        ],
        columns=["Strategy", "Total return", "Annualised return", "Max drawdown",
                 "Volatility", "Sharpe ratio", "Trades", "Days invested"],
    )


def comparison_chart_data(period_results: list[PeriodResult]) -> pd.DataFrame:
    """Annualised return per (period, strategy), for a grouped bar chart.

    Periods where it can't be calculated (zero length) are left out.
    """
    rows = [
        {
            "period": pr.period.name,
            "key": r.strategy,
            "series": strategy_label(r.strategy),
            "annualised_return": r.performance.annualised_return,
            "max_drawdown": -r.performance.max_drawdown.fraction,
        }
        for pr in period_results
        for r in pr.results
        if r.performance.annualised_return is not None
    ]
    return pd.DataFrame(rows, columns=["period", "key", "series", "annualised_return", "max_drawdown"])


# ----- formatting --------------------------------------------------------
# None means "could not be calculated" and is shown as n/a.

def format_money(value) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def format_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0.00%"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
