"""Project-wide settings. Change the stock list or dates here, in one place."""

from datetime import date
from pathlib import Path

# Stocks you can trade. Every one must have been listed by START_DATE.
TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    # Finance
    "JPM", "GS", "MS", "BAC", "V",
    # Other sectors
    "JNJ", "KO", "WMT", "XOM",
]

# SPY is an exchange-traded fund that tracks the S&P 500 index.
BENCHMARK = "SPY"

START_DATE = date(2015, 1, 1)

# Yearly return of a "risk-free" investment (such as US Treasury bills), used by
# the Sharpe ratio. 0.0 is a simplifying assumption; 0.04 would mean 4% a year.
RISK_FREE_RATE = 0.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_DATA_DIR = PROJECT_ROOT / "data" / "prices"
DATABASE_PATH = PROJECT_ROOT / "data" / "simulator.db"
