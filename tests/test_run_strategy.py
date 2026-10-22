"""Tests for running registered strategies and saving them."""

import sqlite3

import pandas as pd
import pytest

from simulator.analytics import compare_saved_run
from simulator.data_loader import DataError
from simulator.run_strategy import main, run_name, run_strategy
from simulator.storage import list_runs, load_snapshots, load_trades


# SPY falls for 220 days, rises for 100, then falls for 100: the 50-day average
# goes below, then above, then below the 200-day average.
CLOSES = (
    [300 - day for day in range(220)]
    + [80 + 2 * day for day in range(1, 101)]
    + [280 - 1.5 * day for day in range(1, 101)]  # ends at 130, still positive
)
DAYS = [d.date() for d in pd.bdate_range("2020-01-02", periods=len(CLOSES))]


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "prices"
    folder.mkdir()
    lines = ["date,close", *(f"{day},{close}" for day, close in zip(DAYS, CLOSES))]
    (folder / "SPY.csv").write_text("\n".join(lines) + "\n")
    return folder


def expected_crossings():
    """Crossing days worked out separately with pandas' own rolling averages."""
    prices = pd.Series(CLOSES, dtype=float)
    short, long = prices.rolling(50).mean(), prices.rolling(200).mean()
    up = (short > long) & (short.shift() <= long.shift())
    down = (short < long) & (short.shift() >= long.shift())
    return [DAYS[i] for i in up[up].index], [DAYS[i] for i in down[down].index]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


def count_rows(db, table):
    with sqlite3.connect(db) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_buy_and_hold_saved_under_its_name(data_dir, db):
    sim = run_strategy("spy_buy_and_hold", data_dir, db)
    assert list_runs(db) == ["spy_buy_and_hold"]
    assert load_trades("spy_buy_and_hold", db) == sim.trades
    assert len(sim.trades) == 1


def test_buy_and_hold_equals_the_benchmark(data_dir, db):
    run_strategy("spy_buy_and_hold", data_dir, db)
    comparison = compare_saved_run("spy_buy_and_hold", db, data_dir)
    assert comparison.strategy.final_value == comparison.benchmark.final_value
    assert comparison.excess_return_pct == 0


def test_crossover_trades_exactly_on_the_50_200_crossing_days(data_dir, db):
    ups, downs = expected_crossings()
    assert len(ups) == 1 and len(downs) == 1 and ups[0] < downs[0]  # the test data works

    sim = run_strategy("spy_sma_crossover_50_200", data_dir, db)
    assert [(t.trade_date, t.side) for t in sim.trades] == [(ups[0], "BUY"), (downs[0], "SELL")]
    assert sim.trades[0].trade_date > DAYS[200]  # no trade before both averages exist
    assert sim.history[-1].holdings == {}  # sold out after the downward cross
    assert load_snapshots("spy_sma_crossover_50_200", db) == sim.history


def test_crossover_never_buys_into_an_existing_uptrend(tmp_path, db):
    """If the 50-day average is already above the 200-day one when both first
    exist, there is no crossing, so no buy (until a later upward cross)."""
    folder = tmp_path / "rising"
    folder.mkdir()
    days = pd.bdate_range("2020-01-02", periods=260)
    lines = ["date,close", *(f"{d.date()},{100 + i}" for i, d in enumerate(days))]
    (folder / "SPY.csv").write_text("\n".join(lines) + "\n")
    sim = run_strategy("spy_sma_crossover_50_200", folder, db)
    assert sim.trades == []


def test_rerunning_replaces_instead_of_duplicating(data_dir, db):
    first = run_strategy("spy_sma_crossover_50_200", data_dir, db)
    before = {table: count_rows(db, table) for table in ("runs", "trades", "snapshots", "holdings")}
    second = run_strategy("spy_sma_crossover_50_200", data_dir, db)
    after = {table: count_rows(db, table) for table in ("runs", "trades", "snapshots", "holdings")}
    assert before == after
    assert first.history == second.history  # same data, same result
    assert list_runs(db) == ["spy_sma_crossover_50_200"]


def test_strategies_are_saved_separately(data_dir, db):
    run_strategy("spy_buy_and_hold", data_dir, db)
    run_strategy("spy_sma_crossover_50_200", data_dir, db)
    assert list_runs(db) == ["spy_buy_and_hold", "spy_sma_crossover_50_200"]


def test_run_without_saving(data_dir, db):
    run_strategy("spy_buy_and_hold", data_dir, db, save=False)
    assert not db.exists()


def test_unknown_strategy(data_dir, db):
    with pytest.raises(ValueError, match="unknown strategy"):
        run_strategy("nope", data_dir, db)


def test_missing_price_data(tmp_path, db):
    with pytest.raises(DataError, match="SPY"):
        run_strategy("spy_buy_and_hold", tmp_path, db)


# ----- next-day execution ----------------------------------------------------

def test_run_names():
    assert run_name("spy_buy_and_hold") == "spy_buy_and_hold"
    assert run_name("spy_buy_and_hold", "next_close") == "spy_buy_and_hold__next_close"


def test_next_close_trades_one_trading_day_after_same_close(data_dir, db):
    same = run_strategy("spy_sma_crossover_50_200", data_dir, db)
    later = run_strategy("spy_sma_crossover_50_200", data_dir, db, execution="next_close")
    next_day = {DAYS[i]: DAYS[i + 1] for i in range(len(DAYS) - 1)}
    assert [(next_day[t.trade_date], t.side) for t in same.trades] == [
        (t.trade_date, t.side) for t in later.trades
    ]
    assert list_runs(db) == ["spy_sma_crossover_50_200", "spy_sma_crossover_50_200__next_close"]


def test_rerunning_next_close_replaces_its_own_copy_only(data_dir, db):
    run_strategy("spy_buy_and_hold", data_dir, db)
    run_strategy("spy_buy_and_hold", data_dir, db, execution="next_close")
    run_strategy("spy_buy_and_hold", data_dir, db, execution="next_close")
    assert list_runs(db) == ["spy_buy_and_hold", "spy_buy_and_hold__next_close"]
    assert count_rows(db, "runs") == 2
    assert count_rows(db, "snapshots") == 2 * len(DAYS)
    [same_trade] = load_trades("spy_buy_and_hold", db)
    [later_trade] = load_trades("spy_buy_and_hold__next_close", db)
    assert (same_trade.trade_date, later_trade.trade_date) == (DAYS[0], DAYS[1])


def test_cli_next_close(data_dir, db, monkeypatch, capsys):
    import simulator.run_strategy as runner

    monkeypatch.setattr(runner, "run_strategy",
                        lambda name, execution: run_strategy(name, data_dir, db, execution=execution))
    assert main(["spy_buy_and_hold", "--execution", "next_close"]) == 0
    assert "spy_buy_and_hold__next_close: 1 trade(s)" in capsys.readouterr().out
    assert list_runs(db) == ["spy_buy_and_hold__next_close"]


# ----- command line ----------------------------------------------------------

def test_cli_list(capsys):
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "spy_buy_and_hold" in output and "spy_sma_crossover_50_200" in output


def test_cli_unknown_name_runs_nothing(capsys):
    assert main(["spy_buy_and_hold", "nope"]) == 1
    output = capsys.readouterr().out
    assert "unknown strategy 'nope'" in output
    assert "saved" not in output  # names are checked before anything runs
