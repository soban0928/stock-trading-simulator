"""Tests for the day-by-day engine, using small made-up price files."""

from datetime import date
from decimal import Decimal

import pytest

from simulator.data_loader import PriceHistory
from simulator.engine import Simulation, SimulationFinishedError
from simulator.portfolio import InsufficientFundsError, InsufficientSharesError

DAY1, DAY2, DAY3, DAY4 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)


@pytest.fixture
def prices(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "date,close\n2024-01-02,10\n2024-01-03,12\n2024-01-04,9\n2024-01-05,15\n"
    )
    (tmp_path / "BBB.csv").write_text(
        "date,close\n2024-01-02,50\n2024-01-03,55\n2024-01-04,60\n2024-01-05,40\n"
    )
    return PriceHistory.from_directory(tmp_path, ["AAA", "BBB"])


@pytest.fixture
def sim(prices):
    """$10,000 starting cash and the default $1 commission."""
    return Simulation(prices)


# ----- starting point --------------------------------------------------------

def test_starts_on_first_day_with_10000_cash(sim):
    assert sim.current_day == DAY1
    assert sim.portfolio.cash == Decimal("10000.00")
    assert sim.history == []


# ----- trading at today's price ----------------------------------------------

def test_buy_uses_todays_close_and_commission(sim):
    trade = sim.buy("AAA", 10)
    assert trade.trade_date == DAY1
    assert trade.price == Decimal("10")
    assert sim.portfolio.cash == Decimal("9899.00")  # 10000 - 100 - 1


def test_ticker_is_case_insensitive(sim):
    sim.buy("aaa", 1)
    assert sim.portfolio.quantity_held("AAA") == Decimal("1")


def test_unknown_ticker(sim):
    with pytest.raises(KeyError):
        sim.buy("ZZZ", 1)


def test_fractional_shares(sim):
    sim.buy("BBB", "0.5")  # 25 + 1 commission
    assert sim.portfolio.quantity_held("BBB") == Decimal("0.5")
    assert sim.portfolio.cash == Decimal("9974.00")


def test_cannot_buy_without_enough_cash(sim):
    with pytest.raises(InsufficientFundsError):
        sim.buy("BBB", 200)  # needs 10,001
    assert sim.portfolio.cash == Decimal("10000.00")
    assert sim.trades == []


def test_cannot_sell_more_than_held(sim):
    sim.buy("AAA", 5)
    with pytest.raises(InsufficientSharesError):
        sim.sell("AAA", 6)
    assert sim.portfolio.quantity_held("AAA") == Decimal("5")


def test_sell_on_a_later_day_uses_that_days_price(sim):
    sim.buy("AAA", 10)  # costs 100 + 1
    sim.end_day()
    trade = sim.sell("AAA", 10)  # day 2 price is 12: 120 - 1
    assert trade.trade_date == DAY2
    assert trade.price == Decimal("12")
    assert trade.realised_pnl == Decimal("18.00")  # 119 - 101
    assert sim.portfolio.cash == Decimal("10018.00")


# ----- daily records ---------------------------------------------------------

def test_end_day_records_snapshot_and_moves_forward(sim):
    sim.buy("AAA", 10)
    snapshot = sim.end_day()
    assert snapshot.day == DAY1
    assert snapshot.cash == Decimal("9899.00")
    assert snapshot.market_value == Decimal("100.00")
    assert snapshot.total_value == Decimal("9999.00")
    assert snapshot.profit_loss == Decimal("-1.00")  # only the commission lost so far
    assert snapshot.holdings == {"AAA": Decimal("10")}
    assert sim.current_day == DAY2


def test_value_follows_each_days_price(sim):
    sim.buy("AAA", 10)
    sim.end_day()
    snapshot = sim.end_day()  # day 2: AAA closes at 12
    assert snapshot.market_value == Decimal("120.00")
    assert snapshot.profit_loss == Decimal("19.00")  # 9899 + 120 - 10000


def test_past_snapshots_do_not_change(sim):
    sim.buy("AAA", 10)
    first = sim.end_day()
    sim.sell("AAA", 10)
    sim.end_day()
    assert first.holdings == {"AAA": Decimal("10")}
    assert sim.history[1].holdings == {}


# ----- no look-ahead ---------------------------------------------------------

def test_past_closes_never_include_future_days(sim):
    assert list(sim.past_closes("AAA")) == [10.0]
    sim.end_day()
    assert list(sim.past_closes("AAA")) == [10.0, 12.0]


def test_run_visits_days_in_order_without_seeing_the_future(sim):
    seen = []

    def strategy(s):
        latest_date_visible = s.past_closes("AAA").index[-1].date()
        seen.append((s.current_day, latest_date_visible))

    sim.run(strategy)
    assert seen == [(DAY1, DAY1), (DAY2, DAY2), (DAY3, DAY3), (DAY4, DAY4)]


def test_cannot_trade_or_advance_after_last_day(sim):
    sim.run(lambda s: None)
    assert sim.finished
    with pytest.raises(SimulationFinishedError):
        sim.buy("AAA", 1)
    with pytest.raises(SimulationFinishedError):
        sim.end_day()


# ----- whole runs ------------------------------------------------------------

def test_buy_and_hold_run(sim):
    def buy_and_hold(s):
        if s.current_day == DAY1:
            s.buy("AAA", 100)  # 1000 + 1

    history = sim.run(buy_and_hold)
    assert [snap.day for snap in history] == [DAY1, DAY2, DAY3, DAY4]
    assert [snap.total_value for snap in history] == [
        Decimal("9999.00"),   # 8999 + 100 * 10
        Decimal("10199.00"),  # 8999 + 100 * 12
        Decimal("9899.00"),   # 8999 + 100 * 9
        Decimal("10499.00"),  # 8999 + 100 * 15
    ]
    assert history[-1].profit_loss == Decimal("499.00")
    assert len(sim.trades) == 1


def test_start_and_end_limit_the_days(prices):
    sim = Simulation(prices, start=DAY2, end=DAY3)
    history = sim.run(lambda s: None)
    assert [snap.day for snap in history] == [DAY2, DAY3]


def test_empty_date_range_is_rejected(prices):
    with pytest.raises(ValueError):
        Simulation(prices, start=date(2030, 1, 1))
