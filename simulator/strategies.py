"""Ready-made trading strategies for the day-by-day engine.

A strategy is anything you can call as strategy(sim). Simulation.run() calls
it once per trading day, oldest day first. On each call a strategy may look
at what the simulation allows (today's price, past_closes, the portfolio) and
then buy or sell at today's closing price. It can never see later prices.

Every strategy here is stateless: it decides from the portfolio and price
history alone, so the same strategy object gives the same result every run.

The registry at the bottom gives each strategy a unique name. That name is also
the run name it is saved under in the database (see simulator.run_strategy).
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from simulator.config import BENCHMARK
from simulator.engine import Simulation
from simulator.portfolio import Trade, affordable_quantity  # noqa: F401  (re-exported)


# ----- shared helpers ----------------------------------------------------
# Both go through sim.submit_order, so they follow the simulation's execution
# setting: filled now at today's close ("same_close"), or at the next trading
# day's close ("next_close"), where the quantity is worked out on that day.

def buy_all(sim: Simulation, ticker: str) -> Trade | None:
    """Spend all available cash on one ticker.

    Returns the trade if it was filled now; None if it was queued for the next
    day, or if the cash can't buy even the smallest fraction of a share.
    """
    return sim.submit_order(ticker, "BUY")


def sell_all(sim: Simulation, ticker: str) -> Trade | None:
    """Sell every share held of one ticker. None if queued or nothing is held."""
    return sim.submit_order(ticker, "SELL")


# ----- strategies --------------------------------------------------------

@dataclass(frozen=True)
class BuyAndHold:
    """Invest all the cash in one ticker at the first chance, then never trade again.

    "First chance" is normally the first day. If the cash can't buy even the
    smallest fraction of a share after commission, it simply tries again the
    next day.
    """

    ticker: str = BENCHMARK

    def __call__(self, sim: Simulation) -> None:
        if not sim.trades:  # nothing bought yet
            buy_all(sim, self.ticker)


@dataclass(frozen=True)
class MovingAverageCrossover:
    """Hold a ticker while its short moving average is above its long one.

    - BUY (all cash) on the day the short average crosses above the long one.
    - SELL (all shares) on the day it crosses below.
    - Do nothing until there are enough prices to calculate both averages
      for today *and* the day before (a crossing compares the two days).

    The averages use closes up to and including today, which is the same
    information the engine uses for the trade price (today's close). No later
    price is ever used.
    """

    ticker: str = BENCHMARK
    short_window: int = 50
    long_window: int = 200

    def __post_init__(self):
        if not (isinstance(self.short_window, int) and isinstance(self.long_window, int)):
            raise ValueError("moving-average windows must be whole numbers of days")
        if not 0 < self.short_window < self.long_window:
            raise ValueError("need 0 < short_window < long_window")

    def signal(self, closes: pd.Series) -> str | None:
        """'BUY', 'SELL' or None, from closing prices ending today (oldest first)."""
        if len(closes) < self.long_window + 1:
            return None  # not enough history for today's and yesterday's long average

        def average(window: int, days_ago: int) -> float:
            end = len(closes) - days_ago
            return float(closes.iloc[end - window:end].mean())

        short_now, long_now = average(self.short_window, 0), average(self.long_window, 0)
        short_before, long_before = average(self.short_window, 1), average(self.long_window, 1)

        if short_before <= long_before and short_now > long_now:
            return "BUY"
        if short_before >= long_before and short_now < long_now:
            return "SELL"
        return None

    def __call__(self, sim: Simulation) -> None:
        signal = self.signal(sim.past_closes(self.ticker))
        if signal == "BUY":
            buy_all(sim, self.ticker)  # does nothing if there is no cash to spend
        elif signal == "SELL":
            sell_all(sim, self.ticker)  # does nothing if no shares are held


# ----- registry ----------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """A named strategy that can be run from the command line and saved."""

    name: str  # unique; also the run name in the database
    description: str
    strategy: Callable[[Simulation], None]
    tickers: list[str] = field(default_factory=list)  # price files it needs


def _registry(*specs: StrategySpec) -> dict[str, StrategySpec]:
    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"strategy names must be unique: {sorted(duplicates)}")
    return {spec.name: spec for spec in specs}


STRATEGIES = _registry(
    StrategySpec(
        name="spy_buy_and_hold",
        description=f"Invest all cash in {BENCHMARK} on the first day and hold it to the end.",
        strategy=BuyAndHold(BENCHMARK),
        tickers=[BENCHMARK],
    ),
    StrategySpec(
        name="spy_sma_crossover_50_200",
        description=(
            f"Hold {BENCHMARK} while its 50-day average is above its 200-day average: "
            "buy all on an upward cross, sell all on a downward cross."
        ),
        strategy=MovingAverageCrossover(BENCHMARK, short_window=50, long_window=200),
        tickers=[BENCHMARK],
    ),
)


def get_strategy(name: str) -> StrategySpec:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from: {', '.join(STRATEGIES)}")
    return STRATEGIES[name]
