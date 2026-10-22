"""Tests for the dashboard's data layer (no Streamlit needed).

Same made-up prices as test_analytics.py: buying 100 AAA on day 1 gives daily
values 9999, 10199, 9899, 10499; SPY buy-and-hold gives 9999.00, 10998.90,
9899.01, 12098.79.
"""

from datetime import date
from decimal import Decimal

import pytest

from dashboard import data
from simulator.data_loader import PriceHistory
from simulator.engine import Simulation
from simulator.storage import RunNotFoundError, save_simulation

DAY1, DAY4 = date(2024, 1, 2), date(2024, 1, 5)


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "prices"
    folder.mkdir()
    (folder / "AAA.csv").write_text(
        "date,close\n2024-01-02,10\n2024-01-03,12\n2024-01-04,9\n2024-01-05,15\n"
    )
    (folder / "SPY.csv").write_text(
        "date,close\n2024-01-02,100\n2024-01-03,110\n2024-01-04,99\n2024-01-05,121\n"
    )
    return folder


@pytest.fixture
def db(tmp_path, data_dir):
    path = tmp_path / "test.db"
    sim = Simulation(PriceHistory.from_directory(data_dir, ["AAA"]))
    sim.run(lambda s: None if s.history else s.buy("AAA", 100))
    save_simulation(sim, "my_run", path)
    return path


@pytest.fixture
def run(db, data_dir):
    return data.load_run("my_run", db, data_dir, risk_free_rate=0.0)


# ----- loading ---------------------------------------------------------------

def test_load_run_with_benchmark(run):
    assert run.name == "my_run"
    assert run.starting_cash == Decimal("10000.00")
    assert len(run.snapshots) == 4
    assert len(run.trades) == 1
    assert run.strategy.final_value == Decimal("10499.00")
    assert run.benchmark.final_value == Decimal("12098.79")
    assert run.benchmark_error is None
    assert run.comparison.excess_return_pct == pytest.approx(0.0499 - 0.209879)


def test_load_run_without_spy_data_still_shows_strategy(db, data_dir):
    (data_dir / "SPY.csv").unlink()
    run = data.load_run("my_run", db, data_dir)
    assert run.strategy.final_value == Decimal("10499.00")
    assert run.benchmark is None and run.benchmark_snapshots is None
    assert run.comparison is None
    assert "SPY" in run.benchmark_error
    assert set(data.equity_curve(run)["key"]) == {"strategy"}
    assert list(data.metrics_table(run)[data.BENCHMARK_LABEL]) == ["n/a"] * 5


def test_load_unknown_run(db, data_dir):
    with pytest.raises(RunNotFoundError):
        data.load_run("nope", db, data_dir)


# ----- chart data ------------------------------------------------------------

def test_equity_curve(run):
    curve = data.equity_curve(run)
    strategy = curve[curve["key"] == "strategy"]
    benchmark = curve[curve["key"] == "benchmark"]
    assert list(strategy["value"]) == [9999.0, 10199.0, 9899.0, 10499.0]
    assert list(benchmark["value"]) == [9999.0, 10998.9, 9899.01, 12098.79]
    assert set(curve["series"]) == {data.STRATEGY_LABEL, data.BENCHMARK_LABEL}
    assert curve["date"].iloc[0].date() == DAY1


def test_drawdown_curve(run):
    curve = data.drawdown_curve(run)
    strategy = curve[curve["key"] == "strategy"]
    # Peaks so far: 10000 (cash), 10000, 10199, 10199, 10499.
    assert list(strategy["drawdown"]) == pytest.approx([-0.0001, 0.0, 9899 / 10199 - 1, 0.0])
    assert len(strategy) == len(run.snapshots)


def test_drawdown_curve_agrees_with_analytics(run):
    curve = data.drawdown_curve(run)
    for key, performance in (("strategy", run.strategy), ("benchmark", run.benchmark)):
        worst = curve[curve["key"] == key]["drawdown"].min()
        assert -worst == pytest.approx(performance.max_drawdown.fraction)


# ----- tables ----------------------------------------------------------------

def test_trades_table(run):
    table = data.trades_table(run.trades)
    assert list(table.columns) == ["Date", "Ticker", "Action", "Quantity", "Price", "Commission"]
    assert table.iloc[0].to_dict() == {
        "Date": DAY1, "Ticker": "AAA", "Action": "BUY",
        "Quantity": 100.0, "Price": 10.0, "Commission": 1.0,
    }


def test_trades_table_empty():
    table = data.trades_table([])
    assert table.empty
    assert "Ticker" in table.columns


def test_history_table_is_newest_first(run):
    table = data.history_table(run.snapshots)
    assert list(table.columns) == ["Date", "Cash", "Market value", "Total value", "Profit/loss"]
    assert table.iloc[0].to_dict() == {
        "Date": DAY4, "Cash": 8999.0, "Market value": 1500.0,
        "Total value": 10499.0, "Profit/loss": 499.0,
    }
    assert table.iloc[-1]["Date"] == DAY1


def test_metrics_table(run):
    table = data.metrics_table(run).set_index("Metric")
    assert list(table.index) == [
        "Total return", "Annualised return", "Max drawdown", "Volatility (annualised)", "Sharpe ratio",
    ]
    assert table.loc["Total return", data.STRATEGY_LABEL] == "+4.99%"
    assert table.loc["Total return", data.BENCHMARK_LABEL] == "+20.99%"
    assert table.loc["Max drawdown", data.STRATEGY_LABEL] == "-2.94%"  # 300 / 10199


def test_strategy_description():
    assert "50-day" in data.strategy_description("spy_sma_crossover_50_200")
    assert "SPY" in data.strategy_description("spy_buy_and_hold")
    assert data.strategy_description("my_run") is None  # not a registered strategy
    assert data.strategy_description("spy_buy_and_hold").endswith("signal day's close.")
    assert data.strategy_description("spy_buy_and_hold__next_close").endswith(
        "next trading day's close."
    )
    assert data.strategy_description("spy_buy_and_hold__unknown_mode") is None


# ----- strategy comparison ---------------------------------------------------

def test_load_comparison_uses_available_periods(data_dir):
    results = data.load_comparison(data_dir)
    # The test prices are all in January 2024, so only these periods have data.
    assert [pr.period.name for pr in results] == ["Full history", "2023–2026"]
    assert [r.strategy for r in results[0].results] == ["spy_buy_and_hold", "spy_sma_crossover_50_200"]


def test_comparison_period_table(data_dir):
    table = data.comparison_period_table(data.load_comparison(data_dir)[0])
    assert list(table["Strategy"]) == ["SPY buy-and-hold (benchmark)", "SMA 50/200 crossover"]
    row = table.iloc[0].to_dict()
    assert row["Total return"] == "+20.99%"  # 99.99 SPY shares x 121 = 12098.79
    assert row["Trades"] == 1 and row["Days invested"] == "100%"
    # 4 days is far too short for 200-day averages: the crossover never trades.
    assert table.iloc[1]["Total return"] == "0.00%"
    assert table.iloc[1]["Sharpe ratio"] == "n/a"


def test_comparison_chart_data(data_dir):
    chart = data.comparison_chart_data(data.load_comparison(data_dir))
    assert set(chart["key"]) == {"spy_buy_and_hold", "spy_sma_crossover_50_200"}
    assert (chart["max_drawdown"] <= 0).all()
    assert len(chart) == 4  # 2 periods x 2 strategies


def test_strategy_labels():
    assert data.strategy_label("spy_buy_and_hold") == "SPY buy-and-hold (benchmark)"
    assert data.strategy_label("something_new") == "something_new"


# ----- formatting ------------------------------------------------------------

def test_format_money():
    assert data.format_money(Decimal("1234.5")) == "$1,234.50"
    assert data.format_money(Decimal("-1234.5")) == "-$1,234.50"
    assert data.format_money(None) == "n/a"


def test_format_pct():
    assert data.format_pct(0.0499) == "+4.99%"
    assert data.format_pct(-0.1) == "-10.00%"
    assert data.format_pct(0.1756, signed=False) == "17.56%"
    assert data.format_pct(0.0) == "0.00%"
    assert data.format_pct(None) == "n/a"


def test_format_ratio():
    assert data.format_ratio(0.7349) == "0.73"
    assert data.format_ratio(None) == "n/a"
