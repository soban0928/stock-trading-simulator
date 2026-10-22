"""Save simulation results to a SQLite database and read them back.

SQLite keeps a whole database in a single file (data/simulator.db by default)
and comes with Python, so nothing extra needs installing.

Each saved simulation is a "run" with a unique name, such as "buy_and_hold".
Its trades and daily snapshots go into their own tables, linked to the run by
run_id.

Money and share quantities are stored as TEXT rather than REAL. REAL is a
float, and floats would undo the exact Decimal arithmetic the portfolio uses.

Run from the project root to list what is saved:
    python -m simulator.storage
"""

import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path

from simulator.config import DATABASE_PATH
from simulator.engine import DailySnapshot, Simulation
from simulator.portfolio import Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    starting_cash TEXT NOT NULL,
    commission    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    run_id       INTEGER NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    trade_number INTEGER NOT NULL,  -- 1, 2, 3... in the order the trades happened
    trade_date   TEXT NOT NULL,     -- YYYY-MM-DD
    ticker       TEXT NOT NULL,
    side         TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity     TEXT NOT NULL,
    price        TEXT NOT NULL,
    commission   TEXT NOT NULL,
    gross_value  TEXT NOT NULL,
    realised_pnl TEXT NOT NULL,
    PRIMARY KEY (run_id, trade_number)
);

CREATE TABLE IF NOT EXISTS snapshots (
    run_id       INTEGER NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    day          TEXT NOT NULL,     -- YYYY-MM-DD
    cash         TEXT NOT NULL,
    market_value TEXT NOT NULL,
    total_value  TEXT NOT NULL,
    profit_loss  TEXT NOT NULL,
    PRIMARY KEY (run_id, day)
);

-- One row per stock held at the end of a day.
CREATE TABLE IF NOT EXISTS holdings (
    run_id   INTEGER NOT NULL,
    day      TEXT NOT NULL,
    ticker   TEXT NOT NULL,
    quantity TEXT NOT NULL,
    PRIMARY KEY (run_id, day, ticker),
    FOREIGN KEY (run_id, day) REFERENCES snapshots (run_id, day) ON DELETE CASCADE
);
"""


class StorageError(Exception):
    """Base class for every database error raised by this module."""


class RunExistsError(StorageError):
    """A run with this name is already saved."""


class RunNotFoundError(StorageError):
    """No run with this name is saved."""


def init_database(db_path: Path = DATABASE_PATH) -> None:
    """Create the database file and its tables if they don't exist yet."""
    with closing(_connect(db_path)):
        pass


def save_simulation(
    sim: Simulation, name: str, db_path: Path = DATABASE_PATH, overwrite: bool = False
) -> int:
    """Save a simulation's trades and daily snapshots under a unique name.

    If the name is already used, raises RunExistsError, unless overwrite=True,
    in which case the old copy is replaced. Returns the new run_id.
    """
    name = _check_name(name)
    if not sim.history:
        raise ValueError("the simulation has no recorded days yet; run it before saving")

    portfolio = sim.portfolio
    # `with connection` makes everything inside one transaction: if anything
    # fails, nothing is saved, so a run is never left half-written.
    with closing(_connect(db_path)) as connection, connection:
        existing = connection.execute("SELECT run_id FROM runs WHERE name = ?", (name,)).fetchone()
        if existing:
            if not overwrite:
                raise RunExistsError(
                    f"a run named {name!r} is already saved; "
                    "use a different name or pass overwrite=True to replace it"
                )
            # ON DELETE CASCADE also removes the run's trades, snapshots and holdings.
            connection.execute("DELETE FROM runs WHERE run_id = ?", existing)

        run_id = connection.execute(
            "INSERT INTO runs (name, starting_cash, commission) VALUES (?, ?, ?)",
            (name, str(portfolio.starting_cash), str(portfolio.commission)),
        ).lastrowid

        connection.executemany(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id, number, trade.trade_date.isoformat(), trade.ticker, trade.side,
                    str(trade.quantity), str(trade.price), str(trade.commission),
                    str(trade.gross_value), str(trade.realised_pnl),
                )
                for number, trade in enumerate(sim.trades, start=1)
            ],
        )
        connection.executemany(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id, snap.day.isoformat(), str(snap.cash), str(snap.market_value),
                    str(snap.total_value), str(snap.profit_loss),
                )
                for snap in sim.history
            ],
        )
        connection.executemany(
            "INSERT INTO holdings VALUES (?, ?, ?, ?)",
            [
                (run_id, snap.day.isoformat(), ticker, str(quantity))
                for snap in sim.history
                for ticker, quantity in snap.holdings.items()
            ],
        )
    return run_id


def load_run_settings(name: str, db_path: Path = DATABASE_PATH) -> tuple[Decimal, Decimal]:
    """The (starting_cash, commission) a saved run used."""
    with closing(_open_existing(db_path)) as connection:
        run_id = _find_run(connection, name)
        starting_cash, commission = connection.execute(
            "SELECT starting_cash, commission FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return Decimal(starting_cash), Decimal(commission)


def list_runs(db_path: Path = DATABASE_PATH) -> list[str]:
    """Names of every saved run, in the order they were saved."""
    with closing(_open_existing(db_path)) as connection:
        rows = connection.execute("SELECT name FROM runs ORDER BY run_id")
        return [name for (name,) in rows]


def load_trades(name: str, db_path: Path = DATABASE_PATH) -> list[Trade]:
    """The trades of a saved run, oldest first, as the same Trade objects the engine made."""
    with closing(_open_existing(db_path)) as connection:
        run_id = _find_run(connection, name)
        rows = connection.execute(
            "SELECT trade_date, ticker, side, quantity, price, commission, gross_value, realised_pnl "
            "FROM trades WHERE run_id = ? ORDER BY trade_number",
            (run_id,),
        )
        return [
            Trade(
                trade_date=date.fromisoformat(trade_date),
                ticker=ticker,
                side=side,
                quantity=Decimal(quantity),
                price=Decimal(price),
                commission=Decimal(commission),
                gross_value=Decimal(gross_value),
                realised_pnl=Decimal(realised_pnl),
            )
            for trade_date, ticker, side, quantity, price, commission, gross_value, realised_pnl in rows
        ]


def load_snapshots(name: str, db_path: Path = DATABASE_PATH) -> list[DailySnapshot]:
    """The daily snapshots of a saved run, oldest first."""
    with closing(_open_existing(db_path)) as connection:
        run_id = _find_run(connection, name)

        holdings_by_day = defaultdict(dict)
        for day, ticker, quantity in connection.execute(
            "SELECT day, ticker, quantity FROM holdings WHERE run_id = ?", (run_id,)
        ):
            holdings_by_day[day][ticker] = Decimal(quantity)

        rows = connection.execute(
            "SELECT day, cash, market_value, total_value, profit_loss "
            "FROM snapshots WHERE run_id = ? ORDER BY day",
            (run_id,),
        )
        return [
            DailySnapshot(
                day=date.fromisoformat(day),
                cash=Decimal(cash),
                market_value=Decimal(market_value),
                total_value=Decimal(total_value),
                profit_loss=Decimal(profit_loss),
                holdings=holdings_by_day[day],
            )
            for day, cash, market_value, total_value, profit_loss in rows
        ]


# ----- helpers -----------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the database and make sure the tables exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    # SQLite ignores REFERENCES / ON DELETE CASCADE unless this is switched on.
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def _open_existing(db_path: Path) -> sqlite3.Connection:
    """Like _connect, but reading from a database that doesn't exist is an error."""
    if not Path(db_path).exists():
        raise StorageError(f"no database at {db_path}; save a simulation first")
    return _connect(db_path)


def _find_run(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(
        "SELECT run_id FROM runs WHERE name = ?", (_check_name(name),)
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"no saved run named {name!r}")
    return row[0]


def _check_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("run name must be a non-empty string")
    return name.strip()


def main() -> int:
    try:
        names = list_runs()
    except StorageError as error:
        print(error)
        return 1
    if not names:
        print(f"No runs saved in {DATABASE_PATH}")
        return 0

    print(f"Saved runs in {DATABASE_PATH}:")
    for name in names:
        last = load_snapshots(name)[-1]
        trade_count = len(load_trades(name))
        print(
            f"  {name}: {trade_count} trade(s), last day {last.day}, "
            f"value ${last.total_value:,}, profit/loss ${last.profit_loss:,}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
