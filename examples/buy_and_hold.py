"""Example: buy 10 AAPL shares on the first day and hold them until the last day.

Run from the project root (download the prices first):
    python -m simulator.download
    python -m examples.buy_and_hold
"""

import sys
from pathlib import Path

from simulator.config import DATABASE_PATH, PRICE_DATA_DIR, TICKERS
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import Simulation
from simulator.storage import save_simulation

TICKER = "AAPL"
SHARES = 10
RUN_NAME = "buy_and_hold"  # name the results are saved under in the database


def buy_and_hold(sim: Simulation) -> None:
    """Called once per trading day. Buys on the first day, then does nothing."""
    if not sim.history:  # no days recorded yet, so today is the first day
        sim.buy(TICKER, SHARES)


def main(data_dir: Path = PRICE_DATA_DIR) -> int:
    try:
        prices = PriceHistory.from_directory(data_dir, TICKERS)
    except DataError as error:
        # The error message already says how to download the data.
        print(f"Could not load price data: {error}")
        return 1

    sim = Simulation(prices)  # $10,000 cash, $1.00 commission per trade
    history = sim.run(buy_and_hold)
    last_day = history[-1]

    print(f"Strategy:         buy {SHARES} {TICKER} on {history[0].day} and hold")
    print(f"Final date:       {last_day.day}")
    print(f"Remaining cash:   ${last_day.cash:,}")
    print(f"Portfolio value:  ${last_day.total_value:,}")
    print(f"Profit/loss:      ${last_day.profit_loss:,}")
    print(f"Number of trades: {len(sim.trades)}")

    # overwrite=True: running the example again replaces the old copy.
    save_simulation(sim, RUN_NAME, overwrite=True)
    print(f"\nSaved as run {RUN_NAME!r} in {DATABASE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
