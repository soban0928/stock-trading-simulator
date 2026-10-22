"""Tests for the trading strategies, using small made-up price files.

The moving-average tests use short windows (2 and 3 days) so every average can
be checked by hand. The 50/200-day version is the same code with bigger numbers.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from simulator.analytics import benchmark_snapshots
from simulator.data_loader import PriceHistory
from simulator.engine import Simulation
from simulator.portfolio import Portfolio
from simulator.strategies import (
    STRATEGIES,
    BuyAndHold,
    MovingAverageCrossover,
    StrategySpec,
    _registry,
    affordable_quantity,
    buy_all,
    get_strategy,
    sell_all,
)

# Closing prices for the 2/3-day crossover. Working, day by day (index from 0):
#   day 3: 2-day avg 10.0,  3-day avg 10.0   (equal: no cross)
#   day 4: 2-day avg 11.5 > 3-day avg 11.0, and day 3 was equal  -> BUY at 13
#   day 5: 14.5 > 13.0;  day 6: 14.0 > 13.67  (still above: no new signal)
#   day 7: 2-day avg 10.0 < 3-day avg 12.0, day 6 was above      -> SELL at 8
#   day 8: 7.0 < 8.67  (still below)
CROSSOVER_CLOSES = [10, 10, 10, 10, 13, 16, 12, 8, 6]
DAYS = [d.date() for d in pd.bdate_range("2024-01-02", periods=len(CROSSOVER_CLOSES))]


def price_history(tmp_path, closes, ticker="SPY"):
    lines = ["date,close", *(f"{day},{close}" for day, close in zip(DAYS, closes))]
    (tmp_path / f"{ticker}.csv").write_text("\n".join(lines) + "\n")
    return PriceHistory.from_directory(tmp_path, [ticker])


def run(prices, strategy, **kwargs):
    sim = Simulation(prices, **kwargs)
    sim.run(strategy)
    return sim


def short_crossover():
    return MovingAverageCrossover("SPY", short_window=2, long_window=3)


# ----- helpers ---------------------------------------------------------------

def test_affordable_quantity_pays_commission_first():
    assert affordable_quantity(Decimal("10000"), Decimal("1"), Decimal("100")) == Decimal("99.99")


def test_affordable_quantity_rounds_down():
    # 9999 / 13 = 769.1538461...; rounding up would overspend.
    assert affordable_quantity(Decimal("10000"), Decimal("1"), Decimal("13")) == Decimal("769.153846")


def test_affordable_quantity_when_commission_uses_all_cash():
    assert affordable_quantity(Decimal("1"), Decimal("1"), Decimal("100")) == 0
    assert affordable_quantity(Decimal("0.50"), Decimal("1"), Decimal("100")) == 0


def test_buy_all_and_sell_all(tmp_path):
    sim = Simulation(price_history(tmp_path, [100, 110]))
    trade = buy_all(sim, "SPY")
    assert trade.quantity == Decimal("99.99")
    assert sim.portfolio.cash == Decimal("0.00")
    assert buy_all(sim, "SPY") is None  # no cash left
    sim.end_day()
    trade = sell_all(sim, "SPY")  # 99.99 x 110 = 10998.90, minus 1 commission
    assert trade.quantity == Decimal("99.99")
    assert sim.portfolio.cash == Decimal("10997.90")
    assert sell_all(sim, "SPY") is None  # nothing left to sell


# ----- buy and hold ----------------------------------------------------------

def test_buy_and_hold_invests_everything_on_first_day(tmp_path):
    sim = run(price_history(tmp_path, [100, 110, 99, 121]), BuyAndHold("SPY"))
    assert len(sim.trades) == 1
    trade = sim.trades[0]
    assert (trade.trade_date, trade.side, trade.price) == (DAYS[0], "BUY", Decimal("100"))
    assert trade.commission == Decimal("1.00")
    assert [snap.cash for snap in sim.history] == [Decimal("0.00")] * 4
    assert all(snap.holdings == {"SPY": Decimal("99.99")} for snap in sim.history)
    assert sim.history[-1].total_value == Decimal("12098.79")  # 99.99 x 121


def test_buy_and_hold_with_fractional_leftover(tmp_path):
    sim = run(price_history(tmp_path, [13, 14]), BuyAndHold("SPY"))
    assert sim.history[0].holdings == {"SPY": Decimal("769.153846")}
    # 769.153846 x 13 = 9998.999998 -> $9,999.00 after rounding to cents.
    assert sim.history[0].cash == Decimal("0.00")


def test_buy_and_hold_cannot_afford_anything(tmp_path):
    prices = price_history(tmp_path, [100, 100, 100])
    sim = run(prices, BuyAndHold("SPY"), portfolio=Portfolio(starting_cash="1", commission="1"))
    assert sim.trades == []
    assert sim.history[-1].cash == Decimal("1.00")


def test_buy_and_hold_matches_analytics_benchmark(tmp_path):
    """The dashboard benchmark and the saved strategy must agree exactly."""
    prices = price_history(tmp_path, [100, 110, 99, 121])
    sim = run(prices, BuyAndHold("SPY"))
    assert benchmark_snapshots(prices, DAYS[:4], 10000, 1) == sim.history


# ----- moving-average signals ------------------------------------------------

def closes(values):
    return pd.Series([float(v) for v in values])


def test_signal_needs_long_window_plus_one_day():
    strategy = short_crossover()
    assert strategy.signal(closes([10, 10, 10])) is None  # 3 days: only today's 3-day average
    assert strategy.signal(closes([])) is None
    assert strategy.signal(closes([10, 10, 10, 10])) is None  # enough data, but no cross


def test_signal_cross_above_and_below():
    strategy = short_crossover()
    assert strategy.signal(closes(CROSSOVER_CLOSES[:5])) == "BUY"  # day 4
    assert strategy.signal(closes(CROSSOVER_CLOSES[:6])) is None  # day 5: still above
    assert strategy.signal(closes(CROSSOVER_CLOSES[:8])) == "SELL"  # day 7
    assert strategy.signal(closes(CROSSOVER_CLOSES[:9])) is None  # day 8: still below


def test_signal_touching_without_crossing_is_not_a_signal():
    # Averages meet (equal) but the short one never goes above.
    assert short_crossover().signal(closes([12, 12, 9, 11])) is None


def test_signal_uses_only_the_latest_prices():
    """Older prices beyond the long window don't change today's averages."""
    strategy = short_crossover()
    assert strategy.signal(closes([500, 1, *CROSSOVER_CLOSES[:5]])) == "BUY"


@pytest.mark.parametrize("short, long", [(0, 3), (3, 3), (5, 3), (-1, 3), (2.5, 3)])
def test_invalid_windows_are_rejected(short, long):
    with pytest.raises(ValueError):
        MovingAverageCrossover("SPY", short_window=short, long_window=long)


def test_default_windows_are_50_and_200():
    strategy = MovingAverageCrossover()
    assert (strategy.ticker, strategy.short_window, strategy.long_window) == ("SPY", 50, 200)


# ----- moving-average execution ----------------------------------------------

def test_crossover_buys_and_sells_on_signal_days(tmp_path):
    sim = run(price_history(tmp_path, CROSSOVER_CLOSES), short_crossover())
    assert [(t.trade_date, t.side, t.price) for t in sim.trades] == [
        (DAYS[4], "BUY", Decimal("13")),
        (DAYS[7], "SELL", Decimal("8")),
    ]
    assert all(t.commission == Decimal("1.00") for t in sim.trades)


def test_crossover_cash_and_holdings(tmp_path):
    sim = run(price_history(tmp_path, CROSSOVER_CLOSES), short_crossover())
    history = sim.history
    # Days 0-3: no signal, all cash.
    assert all(snap.cash == Decimal("10000.00") and snap.holdings == {} for snap in history[:4])
    # Day 4: buy 769.153846 shares at 13 for $9,999.00 + $1 commission.
    assert history[4].cash == Decimal("0.00")
    assert history[4].holdings == {"SPY": Decimal("769.153846")}
    # Day 7: sell at 8: 769.153846 x 8 = 6153.230768 -> $6,153.23, minus $1.
    assert history[7].cash == Decimal("6152.23")
    assert history[7].holdings == {}
    assert history[8].total_value == Decimal("6152.23")  # day 8: still in cash
    assert sim.trades[1].realised_pnl == Decimal("-3847.77")  # 6152.23 - 10000.00


def test_crossover_sell_signal_without_shares_does_nothing(tmp_path):
    # First crossing is downward: day 4, 2-day avg 8.5 < 3-day avg 9.0.
    sim = run(price_history(tmp_path, [10, 10, 10, 10, 7, 4]), short_crossover())
    assert sim.trades == []


def test_crossover_no_trades_when_averages_never_available(tmp_path):
    prices = price_history(tmp_path, CROSSOVER_CLOSES)
    sim = run(prices, MovingAverageCrossover("SPY", short_window=5, long_window=20))
    assert sim.trades == []
    assert sim.history[-1].total_value == Decimal("10000.00")


def test_crossover_buy_without_enough_cash_does_nothing(tmp_path):
    prices = price_history(tmp_path, CROSSOVER_CLOSES)
    sim = run(prices, short_crossover(), portfolio=Portfolio(starting_cash="1", commission="1"))
    assert sim.trades == []


def test_crossover_uses_history_before_the_simulation_start(tmp_path):
    """Prices before the start date are the past, so they may warm up the averages."""
    prices = price_history(tmp_path, CROSSOVER_CLOSES)
    sim = run(prices, short_crossover(), start=DAYS[4])
    assert sim.trades[0].trade_date == DAYS[4]


def test_crossover_ignores_future_prices(tmp_path):
    """Changing prices after day 5 must not change any decision up to day 5."""
    normal = run(price_history(tmp_path / "a", CROSSOVER_CLOSES), short_crossover())
    changed = run(price_history(tmp_path / "b", [*CROSSOVER_CLOSES[:6], 50, 60, 70]), short_crossover())
    up_to_day_5 = lambda sim: [t for t in sim.trades if t.trade_date <= DAYS[5]]  # noqa: E731
    assert up_to_day_5(normal) == up_to_day_5(changed) != []
    assert normal.history[:6] == changed.history[:6]


@pytest.fixture(autouse=True)
def _make_subfolders(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()


def test_strategy_is_stateless_and_repeatable(tmp_path):
    prices = price_history(tmp_path, CROSSOVER_CLOSES)
    strategy = short_crossover()
    assert run(prices, strategy).history == run(prices, strategy).history


# ----- registry --------------------------------------------------------------

def test_registry_names():
    assert list(STRATEGIES) == ["spy_buy_and_hold", "spy_sma_crossover_50_200"]
    for name, spec in STRATEGIES.items():
        assert spec.name == name
        assert spec.description
        assert spec.tickers == ["SPY"]


def test_registry_strategies_are_configured_as_documented():
    crossover = STRATEGIES["spy_sma_crossover_50_200"].strategy
    assert (crossover.ticker, crossover.short_window, crossover.long_window) == ("SPY", 50, 200)
    assert STRATEGIES["spy_buy_and_hold"].strategy == BuyAndHold("SPY")


def test_registry_rejects_duplicate_names():
    spec = StrategySpec("same", "one", BuyAndHold())
    with pytest.raises(ValueError, match="unique"):
        _registry(spec, StrategySpec("same", "two", BuyAndHold()))


def test_get_strategy():
    assert get_strategy("spy_buy_and_hold").name == "spy_buy_and_hold"
    with pytest.raises(ValueError, match="spy_buy_and_hold"):  # lists the valid names
        get_strategy("nope")
