"""Tests for the fair strategy comparison and robustness periods.

Uses small made-up price files. A short 2/3-day crossover is registered for
these tests (monkeypatch), so signals can be checked by hand.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from simulator import strategies
from simulator.analytics import analyse, benchmark_snapshots
from simulator.comparison import (
    BENCHMARK_STRATEGY,
    PERIODS,
    Period,
    compare_strategies,
    format_report,
    invested_fraction,
    main,
    results_table,
    run_period,
    start_note,
)
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import NEXT_CLOSE, DailySnapshot
from simulator.strategies import BuyAndHold, MovingAverageCrossover, StrategySpec

DAYS = [d.date() for d in pd.bdate_range("2019-12-02", periods=40)]  # 2019-12-02 .. 2020-01-24
# Gentle zig-zag so the 2/3-day averages cross several times.
CLOSES = [100 + 5 * ((i // 3) % 2) - i * 0.1 for i in range(len(DAYS))]
DEC = Period("Dec 2019", date(2019, 12, 1), date(2019, 12, 31))
JAN = Period("Jan 2020", date(2020, 1, 1), date(2020, 1, 31))
SHORT = "test_sma_2_3"


def write_prices(folder, closes, days=DAYS):
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["date,close", *(f"{day},{close}" for day, close in zip(days, closes))]
    (folder / "SPY.csv").write_text("\n".join(lines) + "\n")
    return PriceHistory.from_directory(folder, ["SPY"])


@pytest.fixture
def prices(tmp_path):
    return write_prices(tmp_path / "full", CLOSES)


@pytest.fixture(autouse=True)
def short_crossover(monkeypatch):
    """Register a 2/3-day crossover next to the real strategies, for these tests only."""
    spec = StrategySpec(SHORT, "test", MovingAverageCrossover("SPY", 2, 3), ["SPY"])
    monkeypatch.setitem(strategies.STRATEGIES, SHORT, spec)
    return spec


def days_in(period):
    return [d for d in DAYS if period.start <= d <= period.end]


# ----- date alignment and equal conditions -----------------------------------

def test_every_strategy_uses_identical_days(prices):
    for pr in compare_strategies(prices, [SHORT], periods=[DEC, JAN]):
        expected = days_in(pr.period)
        for r in pr.results:
            assert [s.day for s in r.snapshots] == expected
        assert (pr.first_day, pr.last_day) == (expected[0], expected[-1])


def test_benchmark_is_always_included_once_and_first(prices):
    [pr] = compare_strategies(prices, [SHORT], periods=[JAN])
    assert [r.strategy for r in pr.results] == [BENCHMARK_STRATEGY, SHORT]
    [pr] = compare_strategies(prices, [SHORT, BENCHMARK_STRATEGY], periods=[JAN])
    assert [r.strategy for r in pr.results] == [BENCHMARK_STRATEGY, SHORT]


def test_same_starting_cash_and_commission(prices):
    [pr] = compare_strategies(prices, [SHORT], periods=[JAN], starting_cash="5000", commission="2")
    for r in pr.results:
        assert r.performance.starting_value == Decimal("5000.00")
    # Buy-and-hold pays exactly one $2 commission on day 1; the shares are worth what they cost.
    assert pr.result(BENCHMARK_STRATEGY).snapshots[0].total_value == Decimal("4998.00")


def test_benchmark_matches_analytics_benchmark(prices):
    [pr] = compare_strategies(prices, [SHORT], periods=[JAN])
    expected = benchmark_snapshots(prices, days_in(JAN), 10000, 1)
    assert pr.result(BENCHMARK_STRATEGY).snapshots == expected


def test_benchmark_uses_the_same_execution_as_strategies(prices):
    """Under next_close the benchmark also buys a day late, like every strategy."""
    [pr] = compare_strategies(prices, [SHORT], periods=[JAN], execution=NEXT_CLOSE)
    jan = days_in(JAN)
    benchmark = pr.result(BENCHMARK_STRATEGY)
    assert benchmark.snapshots[0].holdings == {}  # day 1: decided, not yet filled
    assert benchmark.snapshots[1].holdings != {}  # day 2: filled
    assert benchmark.performance.first_day == jan[0]


# ----- metrics ---------------------------------------------------------------

def test_metrics_come_from_analytics(prices):
    for pr in compare_strategies(prices, [SHORT], periods=[DEC, JAN], risk_free_rate=0.03):
        for r in pr.results:
            assert r.performance == analyse(r.snapshots, Decimal("10000"), 0.03)


def test_buy_and_hold_metrics_by_hand(tmp_path):
    days = DAYS[:3]
    prices = write_prices(tmp_path / "bh", [100, 110, 121], days)
    [pr] = compare_strategies(prices, [], periods=[Period("All")])
    p = pr.result(BENCHMARK_STRATEGY).performance
    # 9999 / 100 = 99.99 shares; 99.99 x 121 = 12098.79
    assert p.final_value == Decimal("12098.79")
    assert p.total_return_pct == pytest.approx(0.209879)
    assert p.max_drawdown.fraction == pytest.approx(0.0001)  # only the commission dip


def test_invested_fraction():
    def snap(holdings):
        return DailySnapshot(DAYS[0], Decimal(0), Decimal(0), Decimal(0), Decimal(0), holdings)

    assert invested_fraction([snap({}), snap({"SPY": 1}), snap({"SPY": 1}), snap({})]) == 0.5
    assert invested_fraction([]) == 0.0


def test_results_table(prices):
    table = results_table(compare_strategies(prices, [SHORT], periods=[DEC, JAN]))
    assert len(table) == 4  # 2 periods x 2 strategies
    assert list(table["Period"]) == ["Dec 2019", "Dec 2019", "Jan 2020", "Jan 2020"]
    assert (table["Max drawdown"] <= 0).all()
    assert set(table.columns) >= {
        "Total return", "Annualised return", "Max drawdown", "Volatility", "Sharpe ratio",
    }


# ----- period boundaries -----------------------------------------------------

def test_nothing_outside_the_period(prices):
    for execution in ("same_close", NEXT_CLOSE):
        [pr] = compare_strategies(prices, [SHORT], periods=[DEC], execution=execution)
        for r in pr.results:
            assert all(DEC.start <= s.day <= DEC.end for s in r.snapshots)
        assert pr.trading_days == len(days_in(DEC))


def test_each_period_starts_fresh_in_cash(prices):
    for pr in compare_strategies(prices, [SHORT], periods=[DEC, JAN]):
        crossover = pr.result(SHORT)
        assert crossover.performance.starting_value == Decimal("10000")
        first = crossover.snapshots[0]
        assert first.total_value <= Decimal("10000.00")  # at most one commission lost on day 1


def test_history_days_before(prices):
    [dec, jan] = compare_strategies(prices, [SHORT], periods=[DEC, JAN])
    assert dec.history_days_before == 0
    assert jan.history_days_before == len(days_in(DEC))


def test_periods_outside_the_data_are_skipped(prices):
    results = compare_strategies(prices, [SHORT], periods=[Period("2030", date(2030, 1, 1), date(2030, 12, 31)), JAN])
    assert [pr.period.name for pr in results] == ["Jan 2020"]


def test_partial_period_is_clipped_to_the_data(prices):
    [pr] = compare_strategies(prices, [SHORT], periods=[Period("Wide", date(2019, 1, 1), date(2030, 1, 1))])
    assert (pr.first_day, pr.last_day) == (DAYS[0], DAYS[-1])


def test_single_day_period(prices):
    [pr] = compare_strategies(prices, [SHORT], periods=[Period("One day", DAYS[5], DAYS[5])])
    for r in pr.results:
        assert r.performance.annualised_return is None
        assert r.performance.volatility is None
        assert r.performance.sharpe_ratio is None


def test_next_close_order_on_the_period_end_is_not_filled_inside_it(prices, monkeypatch):
    last = days_in(DEC)[-1]
    buy_on_last_day = StrategySpec(
        "buy_last_day", "test",
        lambda sim: sim.submit_order("SPY", "BUY", 1) if sim.current_day == last else None, ["SPY"],
    )
    monkeypatch.setitem(strategies.STRATEGIES, "buy_last_day", buy_on_last_day)
    [pr] = compare_strategies(prices, ["buy_last_day"], periods=[DEC], execution=NEXT_CLOSE)
    assert pr.result("buy_last_day").trades == 0  # the fill day (Jan) is outside the period


def test_default_periods():
    assert [p.name for p in PERIODS] == ["Full history", "2015–2019", "2020–2022", "2023–2026"]
    assert PERIODS[2].start == date(2020, 1, 1) and PERIODS[2].end == date(2022, 12, 31)


# ----- look-ahead bias -------------------------------------------------------

def test_data_after_the_period_is_never_used(tmp_path, prices):
    """Cutting the data off at the period's end gives exactly the same results."""
    cut = [d for d in DAYS if d <= DEC.end]
    truncated = write_prices(tmp_path / "cut", CLOSES[: len(cut)], cut)
    for execution in ("same_close", NEXT_CLOSE):
        [full] = compare_strategies(prices, [SHORT], periods=[DEC], execution=execution)
        [short] = compare_strategies(truncated, [SHORT], periods=[DEC], execution=execution)
        assert [r.snapshots for r in full.results] == [r.snapshots for r in short.results]


def test_changing_future_prices_changes_nothing(tmp_path, prices):
    changed = [c if d <= DEC.end else c * 3 for d, c in zip(DAYS, CLOSES)]
    other = write_prices(tmp_path / "changed", changed)
    [a] = compare_strategies(prices, [SHORT], periods=[DEC])
    [b] = compare_strategies(other, [SHORT], periods=[DEC])
    assert [r.performance for r in a.results] == [r.performance for r in b.results]


def test_warm_up_uses_only_earlier_history(tmp_path, prices):
    """With December's prices available, January's averages are ready on day 1;
    without them, the strategy needs 4 January days before its first signal."""
    jan = days_in(JAN)
    jan_only = write_prices(tmp_path / "jan", CLOSES[len(DAYS) - len(jan):], jan)
    with_history = run_period(prices, JAN, [SHORT]).result(SHORT)
    without = run_period(jan_only, JAN, [SHORT]).result(SHORT)
    assert "warming up" not in with_history.start_note
    assert without.start_note == f"warming up: not enough history for the 3-day average; first possible signal {jan[3]}"


# ----- start notes -----------------------------------------------------------

def test_start_note_states(tmp_path):
    rising = write_prices(tmp_path / "up", [100 + i for i in range(10)], DAYS[:10])
    falling = write_prices(tmp_path / "down", [100 - i for i in range(10)], DAYS[:10])
    strategy = MovingAverageCrossover("SPY", 2, 3)
    assert "already above" in start_note(strategy, rising, DAYS[5])
    assert "below" in start_note(strategy, falling, DAYS[5])
    assert start_note(BuyAndHold(), rising, DAYS[5]) is None


def test_start_note_for_50_200_with_no_history(tmp_path):
    days = [d.date() for d in pd.bdate_range("2015-01-02", periods=260)]
    prices = write_prices(tmp_path / "long", [100 + i * 0.1 for i in range(260)], days)
    note = start_note(MovingAverageCrossover(), prices, days[0])
    assert note.endswith(f"first possible signal {days[200]}")  # the 201st trading day


def test_start_note_when_data_ends_before_warm_up(tmp_path):
    prices = write_prices(tmp_path / "tiny", [100, 101], DAYS[:2])
    assert start_note(MovingAverageCrossover(), prices, DAYS[0]).endswith("after the data ends")


# ----- report ----------------------------------------------------------------

def test_format_report(prices):
    text = format_report(compare_strategies(prices, [SHORT], periods=[DEC, JAN]), "same_close", 0.0)
    assert "Dec 2019: 2019-12-02 to 2019-12-31" in text
    assert "spy_buy_and_hold" in text and SHORT in text
    assert "starts fresh in cash" in text
    assert "do not show that a strategy is" in text


def test_main_reports_missing_data(monkeypatch, capsys):
    import simulator.comparison as comparison

    def missing():
        raise DataError("no price file for SPY")

    monkeypatch.setattr(comparison, "load_prices", missing)
    assert main([]) == 1
    assert "no price file for SPY" in capsys.readouterr().out
