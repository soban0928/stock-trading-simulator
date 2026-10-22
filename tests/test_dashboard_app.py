"""Smoke tests for the Streamlit page, using Streamlit's built-in test runner.

AppTest runs dashboard/app.py without a browser and lets us inspect what it
showed. The database and price folder are pointed at temporary copies, so the
real data/ folder is never used.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from simulator import config
from simulator.data_loader import PriceHistory
from simulator.engine import Simulation
from simulator.storage import save_simulation

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


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
    prices = PriceHistory.from_directory(data_dir, ["AAA"])
    buy = Simulation(prices)
    buy.run(lambda s: None if s.history else s.buy("AAA", 100))
    save_simulation(buy, "buy_aaa", path)
    idle = Simulation(prices)
    idle.run(lambda s: None)
    save_simulation(idle, "all_cash", path)
    return path


@pytest.fixture
def run_app(monkeypatch):
    def run(db_path, data_dir):
        monkeypatch.setattr(config, "DATABASE_PATH", db_path)
        monkeypatch.setattr(config, "PRICE_DATA_DIR", data_dir)
        st.cache_data.clear()
        return AppTest.from_file(APP, default_timeout=30).run()
    return run


def metric_values(at):
    return {metric.label: metric.value for metric in at.metric}


def test_dashboard_shows_latest_run_with_benchmark(run_app, db, data_dir):
    at = run_app(db, data_dir)
    assert not at.exception
    assert not at.error and not at.warning
    assert at.selectbox[0].options == ["buy_aaa", "all_cash"]
    assert at.selectbox[0].value == "all_cash"  # latest saved run is selected
    assert metric_values(at)["Final portfolio value"] == "$10,000.00"


def test_selecting_another_run_updates_the_page(run_app, db, data_dir):
    at = run_app(db, data_dir)
    at.selectbox[0].select("buy_aaa").run()
    assert not at.exception
    values = metric_values(at)
    assert values["Starting cash"] == "$10,000.00"
    assert values["Final portfolio value"] == "$10,499.00"
    assert values["Total profit/loss"] == "$499.00"
    assert values["Total return"] == "+4.99%"
    assert values["Max drawdown"] == "-2.94%"
    assert len(at.dataframe) == 3  # metrics comparison, trades, daily history


def test_registered_strategy_shows_its_description(run_app, db, data_dir):
    from simulator.run_strategy import run_strategy

    run_strategy("spy_sma_crossover_50_200", data_dir, db)
    at = run_app(db, data_dir)
    assert at.selectbox[0].value == "spy_sma_crossover_50_200"
    assert not at.exception
    assert any("50-day average" in caption.value for caption in at.caption)


def open_comparison(at):
    at.segmented_control[0].set_value("Strategy comparison").run()
    return at


def test_comparison_view(run_app, db, data_dir):
    at = open_comparison(run_app(db, data_dir))
    assert not at.exception and not at.error
    assert at.info[0].value.startswith("Each period is a fresh simulation")
    assert len(at.dataframe) == 2  # Full history and 2023–2026 (the test data is in 2024)
    assert list(at.dataframe[0].value["Strategy"]) == ["SPY buy-and-hold (benchmark)", "SMA 50/200 crossover"]


def test_comparison_view_execution_setting(run_app, db, data_dir):
    at = open_comparison(run_app(db, data_dir))
    same_close = at.dataframe[0].value.iloc[0]["Total return"]
    at.segmented_control[1].set_value("next_close").run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[0]["Total return"] != same_close  # benchmark bought a day later


def test_comparison_view_works_without_a_database(run_app, tmp_path, data_dir):
    at = open_comparison(run_app(tmp_path / "missing.db", data_dir))
    assert not at.exception and not at.error
    assert len(at.dataframe) == 2


def test_comparison_view_without_prices(run_app, db, data_dir):
    (data_dir / "SPY.csv").unlink()
    at = open_comparison(run_app(db, data_dir))
    assert not at.exception
    assert "SPY" in at.error[0].value


def test_no_database_shows_clear_error(run_app, tmp_path, data_dir):
    at = run_app(tmp_path / "missing.db", data_dir)
    assert not at.exception
    assert "No saved simulations yet" in at.error[0].value
    assert not at.metric  # nothing else was drawn


def test_missing_spy_data_shows_warning_but_still_works(run_app, db, data_dir):
    (data_dir / "SPY.csv").unlink()
    at = run_app(db, data_dir)
    assert not at.exception
    assert "benchmark unavailable" in at.warning[0].value
    assert metric_values(at)["Final portfolio value"] == "$10,000.00"
