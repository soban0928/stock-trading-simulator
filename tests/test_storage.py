"""Tests for saving and loading simulation results.

Each test uses its own database file in a temporary folder, so the real
data/simulator.db is never touched.
"""

import sqlite3
from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from simulator.data_loader import PriceHistory
from simulator.engine import Simulation
from simulator.portfolio import Portfolio
from simulator.storage import (
    RunExistsError,
    RunNotFoundError,
    StorageError,
    init_database,
    list_runs,
    load_run_settings,
    load_snapshots,
    load_trades,
    save_simulation,
)

DAY1, DAY2, DAY3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)


@pytest.fixture
def prices(tmp_path):
    (tmp_path / "AAA.csv").write_text("date,close\n2024-01-02,10\n2024-01-03,12\n2024-01-04,9\n")
    (tmp_path / "BBB.csv").write_text("date,close\n2024-01-02,50\n2024-01-03,55\n2024-01-04,60\n")
    return PriceHistory.from_directory(tmp_path, ["AAA", "BBB"])


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


def finished_sim(prices):
    """Buy on day 1, partly sell on day 2, hold on day 3."""
    def strategy(sim):
        if sim.current_day == DAY1:
            sim.buy("AAA", 10)
            sim.buy("BBB", "0.5")
        elif sim.current_day == DAY2:
            sim.sell("AAA", 4)

    sim = Simulation(prices)
    sim.run(strategy)
    return sim


def count_rows(db, table):
    with sqlite3.connect(db) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ----- setting up ------------------------------------------------------------

def test_init_database_creates_file_and_tables(db):
    init_database(db)
    assert db.exists()
    assert list_runs(db) == []
    init_database(db)  # running it again is harmless


def test_creates_missing_parent_folder(tmp_path, prices):
    db = tmp_path / "new_folder" / "test.db"
    save_simulation(finished_sim(prices), "run", db)
    assert db.exists()


# ----- saving and loading ----------------------------------------------------

def test_trades_round_trip_exactly(prices, db):
    sim = finished_sim(prices)
    save_simulation(sim, "my_run", db)
    loaded = load_trades("my_run", db)
    assert loaded == sim.trades
    assert [(t.trade_date, t.ticker, t.side, t.quantity) for t in loaded] == [
        (DAY1, "AAA", "BUY", Decimal("10")),
        (DAY1, "BBB", "BUY", Decimal("0.5")),
        (DAY2, "AAA", "SELL", Decimal("4")),
    ]
    assert loaded[0].price == Decimal("10")
    assert loaded[0].commission == Decimal("1.00")


def test_snapshots_round_trip_exactly(prices, db):
    sim = finished_sim(prices)
    save_simulation(sim, "my_run", db)
    loaded = load_snapshots("my_run", db)
    assert loaded == sim.history
    # Day 2: cash 10000 - 101 - 26 + 47 = 9920; 6 AAA at 12 + 0.5 BBB at 55 = 99.50
    assert loaded[1].cash == Decimal("9920.00")
    assert loaded[1].market_value == Decimal("99.50")
    assert loaded[1].total_value == Decimal("10019.50")
    assert loaded[1].profit_loss == Decimal("19.50")
    assert loaded[1].holdings == {"AAA": Decimal("6"), "BBB": Decimal("0.5")}


def test_run_settings_round_trip(prices, db):
    sim = Simulation(prices, Portfolio(starting_cash="5000", commission="2.50"))
    sim.run(lambda s: None)
    save_simulation(sim, "my_run", db)
    assert load_run_settings("my_run", db) == (Decimal("5000.00"), Decimal("2.50"))
    with pytest.raises(RunNotFoundError):
        load_run_settings("nope", db)


def test_money_is_stored_as_exact_text(prices, db):
    save_simulation(finished_sim(prices), "my_run", db)
    with sqlite3.connect(db) as connection:
        stored = connection.execute("SELECT quantity FROM trades WHERE ticker = 'BBB'").fetchone()[0]
    assert stored == "0.5"


def test_run_without_trades_or_holdings(prices, db):
    sim = Simulation(prices)
    sim.run(lambda s: None)
    save_simulation(sim, "do_nothing", db)
    assert load_trades("do_nothing", db) == []
    assert [s.holdings for s in load_snapshots("do_nothing", db)] == [{}, {}, {}]


def test_several_runs_kept_separate(prices, db):
    save_simulation(finished_sim(prices), "first", db)
    idle = Simulation(prices)
    idle.run(lambda s: None)
    save_simulation(idle, "second", db)
    assert list_runs(db) == ["first", "second"]
    assert len(load_trades("first", db)) == 3
    assert load_trades("second", db) == []


def test_data_survives_a_new_connection(prices, db):
    """Saving and loading in separate steps, like restarting the program."""
    sim = finished_sim(prices)
    save_simulation(sim, "my_run", db)
    del sim
    assert len(load_snapshots("my_run", db)) == 3


# ----- duplicates ------------------------------------------------------------

def test_saving_same_name_twice_is_refused(prices, db):
    sim = finished_sim(prices)
    save_simulation(sim, "my_run", db)
    with pytest.raises(RunExistsError, match="overwrite=True"):
        save_simulation(sim, "my_run", db)
    assert count_rows(db, "runs") == 1
    assert count_rows(db, "trades") == 3
    assert count_rows(db, "snapshots") == 3


def test_overwrite_replaces_without_duplicating(prices, db):
    save_simulation(finished_sim(prices), "my_run", db)
    idle = Simulation(prices)
    idle.run(lambda s: None)
    save_simulation(idle, "my_run", db, overwrite=True)
    assert list_runs(db) == ["my_run"]
    assert load_trades("my_run", db) == []
    assert count_rows(db, "trades") == 0
    assert count_rows(db, "snapshots") == 3
    assert count_rows(db, "holdings") == 0  # old holdings removed too


def test_overwrite_with_same_results_keeps_one_copy(prices, db):
    sim = finished_sim(prices)
    save_simulation(sim, "my_run", db)
    save_simulation(sim, "my_run", db, overwrite=True)
    assert count_rows(db, "runs") == 1
    assert count_rows(db, "trades") == 3
    assert load_snapshots("my_run", db) == sim.history


def test_database_itself_rejects_duplicate_snapshot(prices, db):
    save_simulation(finished_sim(prices), "my_run", db)
    with sqlite3.connect(db) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO snapshots VALUES (1, '2024-01-02', '0', '0', '0', '0')"
            )


# ----- error cases -----------------------------------------------------------

def test_loading_unknown_run(prices, db):
    save_simulation(finished_sim(prices), "my_run", db)
    with pytest.raises(RunNotFoundError):
        load_trades("nope", db)
    with pytest.raises(RunNotFoundError):
        load_snapshots("nope", db)


def test_reading_missing_database_does_not_create_it(db):
    with pytest.raises(StorageError, match="no database"):
        list_runs(db)
    assert not db.exists()


@pytest.mark.parametrize("bad_name", ["", "   ", None])
def test_rejects_bad_run_names(prices, db, bad_name):
    with pytest.raises(ValueError):
        save_simulation(finished_sim(prices), bad_name, db)


def test_rejects_simulation_that_has_not_run(prices, db):
    with pytest.raises(ValueError, match="no recorded days"):
        save_simulation(Simulation(prices), "empty", db)
    assert not db.exists()


def test_failed_save_leaves_nothing_behind(prices, db):
    """If saving fails halfway, the transaction is rolled back."""
    sim = finished_sim(prices)
    # Break the last snapshot, so saving fails after the run and trades were written.
    sim.history[-1] = replace(sim.history[-1], day=None)
    with pytest.raises(AttributeError):
        save_simulation(sim, "my_run", db)
    assert list_runs(db) == []
    assert count_rows(db, "trades") == 0
