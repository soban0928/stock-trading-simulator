"""Download daily closing prices from Yahoo Finance and save them as CSV files.

Run from the project root:
    python -m simulator.download

Each ticker is saved to data/prices/<TICKER>.csv with two columns: date, close.
The simulator never talks to the internet itself; it only reads these files.
That keeps it fast, repeatable, and working offline.

Note: yfinance is an unofficial, community-maintained library. Yahoo's data is
for personal use, so the downloaded CSVs are kept out of Git (see .gitignore).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from simulator.config import BENCHMARK, PRICE_DATA_DIR, START_DATE, TICKERS

PRICE_DECIMALS = 4


def clean_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance history table into a simple date/close table.

    With auto_adjust=False, yfinance's "Close" column is adjusted for stock
    splits but not for dividends, which is what we chose to trade at.
    """
    if raw.empty or "Close" not in raw.columns:
        raise ValueError("no closing price data returned")

    cleaned = pd.DataFrame(
        {
            # The index holds timestamps with a time zone; keep only the date.
            "date": pd.to_datetime(raw.index).date,
            "close": raw["Close"].round(PRICE_DECIMALS).to_numpy(),
        }
    )
    cleaned = cleaned.dropna(subset=["close"])
    cleaned = cleaned.drop_duplicates(subset="date", keep="last")
    return cleaned.sort_values("date").reset_index(drop=True)


def fetch_history(ticker: str, start: date) -> pd.DataFrame:
    # Imported here so the rest of the project (and its tests) works even
    # without yfinance installed or without an internet connection.
    import yfinance as yf

    return yf.Ticker(ticker).history(start=start.isoformat(), auto_adjust=False)


def download_all(tickers: list[str], start: date, output_dir: Path) -> list[str]:
    """Download every ticker. Returns the tickers that failed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for ticker in tickers:
        try:
            prices = clean_history(fetch_history(ticker, start))
        except Exception as error:  # one bad ticker shouldn't stop the rest
            print(f"  {ticker:6} FAILED: {error}")
            failed.append(ticker)
            continue

        path = output_dir / f"{ticker}.csv"
        prices.to_csv(path, index=False)
        first, last = prices["date"].iloc[0], prices["date"].iloc[-1]
        print(f"  {ticker:6} {len(prices):5} days  {first} to {last}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical stock prices.")
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=START_DATE,
        help=f"first date to download, YYYY-MM-DD (default {START_DATE})",
    )
    args = parser.parse_args()

    tickers = TICKERS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers from {args.start} into {PRICE_DATA_DIR}")
    failed = download_all(tickers, args.start, PRICE_DATA_DIR)

    if failed:
        print(f"\n{len(failed)} ticker(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll downloads succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
