"""Tests for next-trading-day execution ("next_close").

The rule being tested: an order placed on day t (after seeing day t's close)
is filled at the close of day t+1, never on day t, and never if there is no
day t+1. Prices are made up; every expected number is worked out in comments.
"""

from decimal import Decimal

import pandas as pd
import pytest

from simulator.data_loader import PriceHistory
from simulator.engine import (
    NEXT_CLOSE,
    SAME_CLOSE,
    ExecutionError,
    Order,
    Simulation,
)
from simulator.portfolio import InsufficientFundsError, InvalidOrderError, Portfolio
from simulator.strategies import BuyAndHold, MovingAverageCrossover, buy_all, sell_all

DAYS = [d.date() for d in pd.bdate_range("2024-01-02", periods=9)]
# AAA jumps from 10 to 20 overnight, then falls: good for testing gaps.
AAA = [10, 20, 5, 8, 8, 8, 8, 8, 8]
BBB = [50, 50, 40, 40, 40, 40, 40, 40, 40]
# Same prices as test_strategies.py: the 2/3-day crossover signals BUY on day 4, SELL on day 7.
CROSSOVER = [10, 10, 10, 10, 13, 16, 12, 8, 6]


def make_prices(folder, **closes_by_ticker):
    folder.mkdir(exist_ok=True)
    for ticker, closes in closes_by_ticker.items():
        lines = ["date,close", *(f"{day},{close}" for day, close in zip(DAYS, closes))]
        (folder / f"{ticker}.csv").write_text("\n".join(lines) + "\n")
    return PriceHistory.from_directory(folder, list(closes_by_ticker))


@pytest.fixture
def prices(tmp_path):
    return make_prices(tmp_path / "ab", AAA=AAA, BBB=BBB)


@pytest.fixture
def sim(prices):
    return Simulation(prices, execution=NEXT_CLOSE)


# ----- settings --------------------------------------------------------------

def test_default_is_same_close(prices):
    assert Simulation(prices).execution == SAME_CLOSE


def test_unknown_execution_setting(prices):
    with pytest.raises(ValueError, match="next_close"):
        Simulation(prices, execution="next_open")  # needs opening prices we don't have


# ----- the core rule: day t -> day t+1 ---------------------------------------

def test_order_waits_until_the_next_day(sim):
    assert sim.submit_order("AAA", "BUY", 10) is None  # queued, not filled
    assert sim.trades == []
    assert sim.portfolio.cash == Decimal("10000.00")
    assert sim.pending_orders == [Order("AAA", "BUY", Decimal("10"), DAYS[0])]

    day_0 = sim.end_day()
    assert day_0.holdings == {} and day_0.cash == Decimal("10000.00")  # nothing on day t

    trade = sim.trades[0]  # filled as the simulation arrives at day 1
    assert (trade.trade_date, trade.price) == (DAYS[1], Decimal("20"))
    assert sim.portfolio.cash == Decimal("9799.00")  # 10000 - 10 x 20 - 1 commission
    assert sim.pending_orders == []


@pytest.mark.parametrize("signal_day", range(len(DAYS) - 1))
def test_signal_on_any_day_fills_exactly_one_day_later(prices, signal_day):
    sim = Simulation(prices, execution=NEXT_CLOSE)

    def strategy(s):
        if s.current_day == DAYS[signal_day]:
            s.submit_order("AAA", "BUY", 1)
            assert s.trades == []  # still not filled while the strategy is running on day t

    sim.run(strategy)
    assert [t.trade_date for t in sim.trades] == [DAYS[signal_day + 1]]
    assert sim.trades[0].price == Decimal(AAA[signal_day + 1])


def test_order_on_the_last_day_is_never_filled(prices):
    sim = Simulation(prices, execution=NEXT_CLOSE)
    sim.run(lambda s: s.submit_order("AAA", "BUY", 1) if s.current_day == DAYS[-1] else None)
    assert sim.trades == []
    assert sim.unfilled_orders == [Order("AAA", "BUY", Decimal("1"), DAYS[-1])]
    assert sim.history[-1].cash == Decimal("10000.00")


def test_unfilled_orders_is_empty_while_running(sim):
    sim.submit_order("AAA", "BUY", 1)
    assert sim.unfilled_orders == []  # still pending, not unfilled


def test_same_close_fills_immediately(prices):
    sim = Simulation(prices)  # default same_close
    trade = sim.submit_order("AAA", "BUY", 10)
    assert (trade.trade_date, trade.price) == (DAYS[0], Decimal("10"))
    assert sim.pending_orders == []


# ----- sizing, cash and commission at fill time ------------------------------

def test_buy_all_is_sized_with_the_fill_days_price(sim):
    buy_all(sim, "AAA")  # day 0 close is 10; sizing at 10 would buy 999.9 shares
    sim.end_day()  # fills at day 1 close of 20
    trade = sim.trades[0]
    assert trade.quantity == Decimal("499.95")  # (10000 - 1) / 20
    assert sim.portfolio.cash == Decimal("0.00")  # never negative after a gap up


def test_fixed_quantity_order_rejected_after_a_gap_up(sim):
    sim.submit_order("AAA", "BUY", 900)  # 9,001 at 10 would be fine...
    sim.end_day()  # ...but at 20 it costs 18,001
    assert sim.trades == []
    assert sim.portfolio.cash == Decimal("10000.00")  # no commission charged either
    [rejected] = sim.rejected_orders
    assert rejected.day == DAYS[1]
    assert rejected.order.quantity == Decimal("900")
    assert "cash is $10000.00" in rejected.reason


def test_rejection_does_not_stop_the_simulation(prices):
    sim = Simulation(prices, execution=NEXT_CLOSE)

    def strategy(s):
        if s.current_day == DAYS[0]:
            s.submit_order("AAA", "BUY", 900)  # rejected on day 1
        elif s.current_day == DAYS[1]:
            s.submit_order("AAA", "BUY", 900)  # filled on day 2 at 5: 4500 + 1

    sim.run(strategy)
    assert len(sim.rejected_orders) == 1
    assert [(t.trade_date, t.quantity) for t in sim.trades] == [(DAYS[2], Decimal("900"))]
    assert sim.history[2].cash == Decimal("5499.00")


def test_commission_is_charged_once_at_fill(sim):
    sim.submit_order("BBB", "BUY", 2)
    sim.end_day()
    assert sim.trades[0].commission == Decimal("1.00")
    assert sim.portfolio.cash == Decimal("9899.00")  # 10000 - 2 x 50 - 1


def test_fractional_shares_at_fill(sim):
    sim.submit_order("BBB", "BUY", "0.5")
    sim.end_day()
    assert sim.portfolio.quantity_held("BBB") == Decimal("0.5")
    assert sim.portfolio.cash == Decimal("9974.00")  # 10000 - 25 - 1


def test_sell_all_uses_holdings_on_the_fill_day(sim):
    sim.submit_order("AAA", "BUY", 10)
    sim.end_day()  # day 1: bought 10 at 20
    sell_all(sim, "AAA")
    assert sim.portfolio.quantity_held("AAA") == Decimal("10")  # still held on day 1
    sim.end_day()  # day 2: sold 10 at 5
    trade = sim.trades[-1]
    assert (trade.trade_date, trade.side, trade.quantity) == (DAYS[2], "SELL", Decimal("10"))
    assert sim.portfolio.cash == Decimal("9848.00")  # 9799 + 50 - 1
    assert trade.realised_pnl == Decimal("-152.00")  # 49 - 201


def test_sells_fill_before_buys_so_proceeds_can_be_reused(sim):
    buy_all(sim, "AAA")
    sim.end_day()  # day 1: 499.95 AAA at 20, cash 0
    buy_all(sim, "BBB")  # placed first...
    sell_all(sim, "AAA")  # ...but the sell must fill first, or there's no cash
    sim.end_day()  # day 2: AAA at 5, BBB at 40
    assert [t.side for t in sim.trades[1:]] == ["SELL", "BUY"]
    # Sell: 499.95 x 5 = 2499.75 - 1 = 2498.75. Buy: (2498.75 - 1) / 40 = 62.44375 shares.
    assert sim.portfolio.quantity_held("BBB") == Decimal("62.44375")
    assert sim.portfolio.cash == Decimal("0.00")


def test_buy_all_without_enough_cash_is_skipped(prices):
    sim = Simulation(prices, Portfolio(starting_cash="1", commission="1"), execution=NEXT_CLOSE)
    buy_all(sim, "AAA")
    sim.end_day()
    assert sim.trades == [] and sim.rejected_orders == []


# ----- guard rails -----------------------------------------------------------

def test_direct_buy_and_sell_are_blocked_in_next_close(sim):
    with pytest.raises(ExecutionError, match="submit_order"):
        sim.buy("AAA", 1)
    with pytest.raises(ExecutionError):
        sim.sell("AAA", 1)


@pytest.mark.parametrize("ticker, side, quantity, error", [
    ("AAA", "HOLD", 1, InvalidOrderError),
    ("AAA", "BUY", 0, InvalidOrderError),
    ("AAA", "BUY", "-1", InvalidOrderError),
    ("AAA", "BUY", "abc", InvalidOrderError),
    ("ZZZ", "BUY", 1, KeyError),
])
def test_bad_orders_fail_when_placed_not_when_filled(sim, ticker, side, quantity, error):
    with pytest.raises(error):
        sim.submit_order(ticker, side, quantity)
    assert sim.pending_orders == []


def test_same_close_fixed_quantity_still_raises(prices):
    """Same-day behaviour is unchanged: a fixed order you can't afford raises at once."""
    with pytest.raises(InsufficientFundsError):
        Simulation(prices).submit_order("AAA", "BUY", 2000)


# ----- strategies under next_close -------------------------------------------

def test_crossover_signals_execute_one_day_later(tmp_path):
    prices = make_prices(tmp_path / "x", SPY=CROSSOVER)
    strategy = MovingAverageCrossover("SPY", short_window=2, long_window=3)
    same = Simulation(prices)
    same.run(strategy)
    later = Simulation(prices, execution=NEXT_CLOSE)
    later.run(strategy)

    assert [(t.trade_date, t.side) for t in same.trades] == [(DAYS[4], "BUY"), (DAYS[7], "SELL")]
    assert [(t.trade_date, t.side, t.price) for t in later.trades] == [
        (DAYS[5], "BUY", Decimal("16")),  # signal day 4 -> filled day 5
        (DAYS[8], "SELL", Decimal("6")),  # signal day 7 -> filled day 8
    ]
    # Buy (10000 - 1) / 16 = 624.9375 shares; sell at 6: 3749.625 -> $3,749.63, minus $1.
    assert later.history[-1].cash == Decimal("3748.63")
    assert same.history[-1].cash == Decimal("6152.23")  # the same-day result, for comparison


def test_crossover_signal_uses_only_prices_up_to_the_signal_day(tmp_path):
    """Changing the fill-day price changes the fill price, never whether there is a signal."""
    strategy = MovingAverageCrossover("SPY", short_window=2, long_window=3)
    changed = [*CROSSOVER[:5], 1000, *CROSSOVER[6:]]  # day 5 (the fill day) is wildly different
    sim = Simulation(make_prices(tmp_path / "y", SPY=changed), execution=NEXT_CLOSE)
    sim.run(strategy)
    first = sim.trades[0]
    assert (first.trade_date, first.side, first.price) == (DAYS[5], "BUY", Decimal("1000"))


def test_buy_and_hold_next_close(tmp_path):
    sim = Simulation(make_prices(tmp_path / "z", SPY=[100, 110, 99]), execution=NEXT_CLOSE)
    sim.run(BuyAndHold("SPY"))
    [trade] = sim.trades
    assert (trade.trade_date, trade.price) == (DAYS[1], Decimal("110"))
    assert trade.quantity == Decimal("90.900000")  # 9999 / 110 = 90.9, exactly
    assert sim.history[0].holdings == {}  # day 0: still in cash
    assert sim.history[-1].total_value == Decimal("8999.10")  # 90.9 x 99
