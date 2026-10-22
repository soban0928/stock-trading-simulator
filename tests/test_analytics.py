"""Tests for performance analytics.

Expected values are worked out by hand in the comments, using small made-up
prices, so each calculation can be checked with a calculator.
"""

import math
from datetime import date
from decimal import Decimal

import pytest

from simulator.analytics import (
    Drawdown,
    analyse,
    annualised_return,
    benchmark_snapshots,
    compare,
    compare_saved_run,
    daily_returns,
    drawdown_series,
    format_report,
    max_drawdown,
    sharpe_ratio,
    total_return_pct,
    volatility,
)
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import Simulation
from simulator.storage import save_simulation

DAY1, DAY2, DAY3, DAY4 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)
D = [date(2024, 1, d) for d in range(1, 8)]  # plain list of dates for pure-function tests


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "date,close\n2024-01-02,10\n2024-01-03,12\n2024-01-04,9\n2024-01-05,15\n"
    )
    (tmp_path / "SPY.csv").write_text(
        "date,close\n2024-01-02,100\n2024-01-03,110\n2024-01-04,99\n2024-01-05,121\n"
    )
    return tmp_path


@pytest.fixture
def prices(data_dir):
    return PriceHistory.from_directory(data_dir, ["AAA"])


@pytest.fixture
def spy(data_dir):
    return PriceHistory.from_directory(data_dir, ["SPY"])


def buy_and_hold_aaa(prices, **dates):
    """Buy 100 AAA on the first day (cost 1000 + 1 commission) and hold.

    Daily values: 9999, 10199, 9899, 10499 (8999 cash + 100 x price).
    """
    sim = Simulation(prices, **dates)
    sim.run(lambda s: None if s.history else s.buy("AAA", 100))
    return sim


# ----- total return ----------------------------------------------------------

def test_total_return_pct():
    assert total_return_pct(10000, 11000) == pytest.approx(0.10)
    assert total_return_pct(Decimal("10000"), Decimal("9000")) == pytest.approx(-0.10)


def test_total_return_pct_rejects_zero_start():
    with pytest.raises(ValueError):
        total_return_pct(0, 100)


# ----- annualised return -----------------------------------------------------

def test_annualised_return_over_two_years():
    # 100 -> 121 is 10% a year for two years (1.1 x 1.1 = 1.21).
    assert annualised_return(100, 121, date(2020, 1, 1), date(2022, 1, 1)) == pytest.approx(0.10, abs=1e-3)


def test_annualised_return_loss():
    # 100 -> 81 is -10% a year for two years (0.9 x 0.9 = 0.81).
    assert annualised_return(100, 81, date(2020, 1, 1), date(2022, 1, 1)) == pytest.approx(-0.10, abs=1e-3)


def test_annualised_return_zero_length_period_is_none():
    assert annualised_return(100, 110, DAY1, DAY1) is None


def test_annualised_return_everything_lost():
    assert annualised_return(100, 0, date(2020, 1, 1), date(2021, 1, 1)) == -1.0


def test_annualised_return_too_large_is_none():
    # x10,000 in one day would be 10,000 ** 365 a year: too big for a float.
    assert annualised_return(100, 1_000_000, DAY1, DAY2) is None


# ----- daily returns ---------------------------------------------------------

def test_daily_returns():
    assert daily_returns([100, 110, 99]) == pytest.approx([0.10, -0.10])


def test_daily_returns_need_two_values():
    assert daily_returns([100]) == []


def test_daily_returns_final_zero_is_allowed():
    assert daily_returns([100, 0]) == [-1.0]


def test_daily_returns_zero_before_end_is_rejected():
    with pytest.raises(ValueError, match="zero"):
        daily_returns([100, 0, 50])


def test_daily_returns_negative_is_rejected():
    with pytest.raises(ValueError):
        daily_returns([100, -5])


# ----- max drawdown ----------------------------------------------------------

def test_max_drawdown_finds_largest_fall():
    # Falls: 120 -> 90 is 25%; 130 -> 117 is only 10%.
    result = max_drawdown([100, 120, 90, 130, 117], D[:5])
    assert result.fraction == pytest.approx(0.25)
    assert (result.peak_day, result.trough_day) == (D[1], D[2])


def test_max_drawdown_uses_only_earlier_peaks():
    # The 200 comes after the low, so it must not count as the peak for 50.
    result = max_drawdown([100, 50, 200], D[:3])
    assert result.fraction == pytest.approx(0.50)
    assert result.peak_day == D[0]


def test_max_drawdown_when_value_never_falls():
    assert max_drawdown([100, 110, 120], D[:3]) == Drawdown(0.0, None, None)


def test_max_drawdown_rejects_mismatched_input():
    with pytest.raises(ValueError):
        max_drawdown([], [])
    with pytest.raises(ValueError):
        max_drawdown([100, 110], D[:1])


# ----- drawdown series -------------------------------------------------------

def test_drawdown_series():
    assert drawdown_series([100, 120, 90, 130, 117]) == pytest.approx([0, 0, -0.25, 0, -0.10])


def test_drawdown_series_matches_max_drawdown():
    values = [100, 50, 200, 150, 210, 105]
    assert -min(drawdown_series(values)) == pytest.approx(max_drawdown(values, D[:6]).fraction)


def test_drawdown_series_edge_cases():
    assert drawdown_series([]) == []
    assert drawdown_series([100]) == [0.0]
    assert drawdown_series([0, 0]) == [0.0, 0.0]  # no division by zero


# ----- volatility ------------------------------------------------------------

def test_volatility_is_yearly_sample_stdev():
    # Sample stdev of [+1%, -1%] is sqrt(0.01^2 + 0.01^2) = 0.01 x sqrt(2).
    assert volatility([0.01, -0.01]) == pytest.approx(0.01 * math.sqrt(2) * math.sqrt(252))


def test_volatility_needs_two_returns():
    assert volatility([]) is None
    assert volatility([0.05]) is None


def test_volatility_of_constant_returns_is_zero():
    assert volatility([0.01, 0.01, 0.01]) == 0.0


# ----- Sharpe ratio ----------------------------------------------------------

def test_sharpe_ratio_with_zero_risk_free_rate():
    # Mean 2%, sample stdev 0.01 x sqrt(2): 0.02 / 0.014142 x sqrt(252).
    assert sharpe_ratio([0.01, 0.03], 0.0) == pytest.approx(math.sqrt(2) * math.sqrt(252))


def test_sharpe_ratio_subtracts_risk_free_rate():
    daily_rf = 1.04 ** (1 / 252) - 1  # 4% a year, compounded daily
    expected = (0.02 - daily_rf) / (0.01 * math.sqrt(2)) * math.sqrt(252)
    assert sharpe_ratio([0.01, 0.03], 0.04) == pytest.approx(expected)
    assert sharpe_ratio([0.01, 0.03], 0.04) < sharpe_ratio([0.01, 0.03], 0.0)


def test_sharpe_ratio_undefined_cases():
    assert sharpe_ratio([0.01]) is None  # too few returns
    assert sharpe_ratio([0.0, 0.0, 0.0]) is None  # zero volatility (all cash)


# ----- analysing a whole run -------------------------------------------------

def test_analyse_buy_and_hold(prices):
    sim = buy_and_hold_aaa(prices)
    result = analyse(sim.history, sim.portfolio.starting_cash)
    assert (result.first_day, result.last_day) == (DAY1, DAY4)
    assert result.starting_value == Decimal("10000.00")
    assert result.final_value == Decimal("10499.00")
    assert result.total_return == Decimal("499.00") == sim.history[-1].profit_loss
    assert result.total_return_pct == pytest.approx(0.0499)
    # First return is the commission: 9999 / 10000 - 1.
    assert result.daily_returns == pytest.approx(
        [-0.0001, 10199 / 9999 - 1, 9899 / 10199 - 1, 10499 / 9899 - 1]
    )
    # Biggest fall: 10199 (day 2) down to 9899 (day 3).
    assert result.max_drawdown.fraction == pytest.approx(300 / 10199)
    assert (result.max_drawdown.peak_day, result.max_drawdown.trough_day) == (DAY2, DAY3)
    assert result.volatility == pytest.approx(volatility(result.daily_returns))
    assert result.sharpe_ratio == pytest.approx(sharpe_ratio(result.daily_returns, 0.0))
    # 3 calendar days: 1.0499 ** (365.25 / 3) - 1
    assert result.annualised_return == pytest.approx(1.0499 ** (365.25 / 3) - 1)


def test_analyse_single_day(prices):
    sim = buy_and_hold_aaa(prices, end=DAY1)
    result = analyse(sim.history, sim.portfolio.starting_cash)
    assert result.total_return == Decimal("-1.00")  # just the commission
    assert result.annualised_return is None
    assert result.volatility is None
    assert result.sharpe_ratio is None


def test_analyse_all_cash(prices):
    sim = Simulation(prices)
    sim.run(lambda s: None)
    result = analyse(sim.history, sim.portfolio.starting_cash)
    assert result.total_return == Decimal("0")
    assert result.volatility == 0.0
    assert result.sharpe_ratio is None
    assert result.max_drawdown.fraction == 0.0


def test_analyse_rejects_bad_input(prices):
    sim = buy_and_hold_aaa(prices)
    with pytest.raises(ValueError, match="no snapshots"):
        analyse([], 10000)
    with pytest.raises(ValueError, match="positive"):
        analyse(sim.history, 0)
    with pytest.raises(ValueError, match="date order"):
        analyse(list(reversed(sim.history)), 10000)


# ----- benchmark -------------------------------------------------------------

def test_benchmark_buys_spy_on_first_day_and_holds(spy):
    # 10000 - 1 commission = 9999 to spend; at 100 that buys 99.99 shares.
    snapshots = benchmark_snapshots(spy, [DAY1, DAY2, DAY3, DAY4], 10000, 1)
    assert snapshots[0].holdings == {"SPY": Decimal("99.99")}
    assert snapshots[0].cash == Decimal("0.00")
    assert [s.total_value for s in snapshots] == [
        Decimal("9999.00"),   # 99.99 x 100
        Decimal("10998.90"),  # 99.99 x 110
        Decimal("9899.01"),   # 99.99 x 99
        Decimal("12098.79"),  # 99.99 x 121
    ]


def test_benchmark_rounds_shares_down_so_it_never_overspends(tmp_path):
    (tmp_path / "SPY.csv").write_text("date,close\n2024-01-02,333.33\n")
    spy = PriceHistory.from_directory(tmp_path, ["SPY"])
    snapshot = benchmark_snapshots(spy, [DAY1], 10000, 1)[0]
    # 9999 / 333.33 = 29.99729997..., rounded down to 6 decimal places.
    assert snapshot.holdings == {"SPY": Decimal("29.997299")}
    assert Decimal("0") <= snapshot.cash < Decimal("0.01")


def test_benchmark_uses_only_the_strategys_days(spy):
    snapshots = benchmark_snapshots(spy, [DAY1, DAY3], 10000, 1)
    assert [s.day for s in snapshots] == [DAY1, DAY3]


def test_benchmark_missing_spy_day_is_an_error(tmp_path):
    (tmp_path / "SPY.csv").write_text("date,close\n2024-01-02,100\n2024-01-04,99\n")
    spy = PriceHistory.from_directory(tmp_path, ["SPY"])
    with pytest.raises(DataError, match="2024-01-03"):
        benchmark_snapshots(spy, [DAY1, DAY2, DAY3], 10000, 1)


def test_benchmark_with_no_spy_data_in_range(spy):
    with pytest.raises(DataError, match="no SPY prices"):
        benchmark_snapshots(spy, [date(2030, 1, 2)], 10000, 1)


# ----- comparison ------------------------------------------------------------

def test_compare_strategy_with_spy(prices, spy):
    sim = buy_and_hold_aaa(prices)
    comparison = compare(sim.history, spy, 10000, 1)
    assert comparison.strategy.total_return_pct == pytest.approx(0.0499)
    assert comparison.benchmark.total_return_pct == pytest.approx(0.209879)  # 12098.79 / 10000 - 1
    assert comparison.excess_return_pct == pytest.approx(0.0499 - 0.209879)
    assert comparison.excess_annualised_return < 0


def test_compare_saved_run_and_report(prices, data_dir, tmp_path):
    db = tmp_path / "test.db"
    save_simulation(buy_and_hold_aaa(prices), "my_run", db)
    comparison = compare_saved_run("my_run", db, data_dir)
    assert comparison.benchmark.final_value == Decimal("12098.79")

    report = format_report("my_run", comparison, 0.0)
    assert "$10,499.00" in report
    assert "$12,098.79" in report
    assert "trailed SPY by 16.00 percentage points" in report
    assert "less than a year" in report


def test_report_shows_na_for_undefined_metrics(prices, spy):
    sim = buy_and_hold_aaa(prices, end=DAY1)
    report = format_report("one_day", compare(sim.history, spy, 10000, 1))
    assert "n/a" in report
