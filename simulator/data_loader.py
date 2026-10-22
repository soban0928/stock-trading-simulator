"""Load and validate the price CSV files produced by simulator.download.

Data from the internet can be incomplete or wrong, so every file is checked
before the simulator trusts it. A clear error now is better than a silently
wrong portfolio value later.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"date", "close"}


class DataError(Exception):
    """A price file is missing or contains invalid data."""


def load_price_file(path: Path) -> pd.Series:
    """Read one CSV and return its closing prices as a Series indexed by date."""
    if not path.exists():
        raise DataError(f"price file not found: {path}")

    try:
        table = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise DataError(f"{path.name}: could not read CSV ({error})")

    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise DataError(f"{path.name}: missing column(s) {sorted(missing)}")
    if table.empty:
        raise DataError(f"{path.name}: file has no rows")

    try:
        dates = pd.to_datetime(table["date"], format="%Y-%m-%d")
    except (ValueError, TypeError):
        raise DataError(f"{path.name}: dates must be in YYYY-MM-DD format")

    closes = pd.to_numeric(table["close"], errors="coerce")
    if closes.isna().any():
        bad_row = int(closes.isna().idxmax()) + 2  # +2: header line, 1-based rows
        raise DataError(f"{path.name}: missing or non-numeric price on line {bad_row}")
    if (closes <= 0).any():
        raise DataError(f"{path.name}: prices must be positive")
    if dates.duplicated().any():
        raise DataError(f"{path.name}: duplicate date {dates[dates.duplicated()].iloc[0].date()}")

    return pd.Series(closes.to_numpy(), index=dates, name=path.stem).sort_index()


class PriceHistory:
    """Closing prices for several tickers on a shared set of trading days.

    Only days where *every* ticker has a price are kept. That way, on any
    trading day, every stock can be bought, sold and valued.
    """

    def __init__(self, table: pd.DataFrame):
        if table.empty:
            raise DataError("no trading days where every ticker has a price")
        self._table = table

    @classmethod
    def from_directory(cls, data_dir: Path, tickers: list[str]) -> "PriceHistory":
        missing = [t for t in tickers if not (data_dir / f"{t}.csv").exists()]
        if missing:
            raise DataError(
                f"no price file for {', '.join(missing)} in {data_dir}. "
                "Run: python -m simulator.download"
            )
        series = [load_price_file(data_dir / f"{ticker}.csv") for ticker in tickers]
        # join="inner" keeps only dates that appear in every file.
        table = pd.concat(series, axis=1, join="inner").sort_index()
        return cls(table)

    @property
    def tickers(self) -> list[str]:
        return list(self._table.columns)

    @property
    def trading_days(self) -> list[date]:
        return [timestamp.date() for timestamp in self._table.index]

    def price(self, ticker: str, day: date) -> Decimal:
        """Closing price of one ticker on one day, as an exact Decimal."""
        if ticker not in self._table.columns:
            raise KeyError(f"unknown ticker: {ticker}")
        timestamp = pd.Timestamp(day)
        if timestamp not in self._table.index:
            raise KeyError(f"{day} is not a trading day in the data")
        # str() first so the Decimal matches the 4-decimal value in the CSV.
        return Decimal(str(self._table.at[timestamp, ticker]))

    def prices_on(self, day: date) -> dict[str, Decimal]:
        return {ticker: self.price(ticker, day) for ticker in self.tickers}

    def closes_through(self, ticker: str, day: date) -> pd.Series:
        """Closing prices of one ticker from the first day up to and including `day`.

        Nothing after `day` is returned, so a strategy cannot peek at the future.
        """
        if ticker not in self._table.columns:
            raise KeyError(f"unknown ticker: {ticker}")
        # .copy() so changing the result cannot change the stored prices.
        return self._table.loc[: pd.Timestamp(day), ticker].copy()
