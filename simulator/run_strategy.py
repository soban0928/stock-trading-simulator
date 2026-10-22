"""Run a named strategy over the downloaded prices and save it to the database.

Run from the project root (download the prices first):
    python -m simulator.run_strategy --list                     # show strategies
    python -m simulator.run_strategy spy_sma_crossover_50_200   # run one
    python -m simulator.run_strategy                            # run them all
    python -m simulator.run_strategy --execution next_close     # fill orders a day later

Each strategy is saved under its own name. Running it again replaces the old
copy (overwrite=True), so the database always holds one copy per strategy.
With --execution next_close the run name gets a "__next_close" suffix, so both
versions can be saved and compared side by side.
"""

import argparse
import sys
from pathlib import Path

from simulator.config import DATABASE_PATH, PRICE_DATA_DIR
from simulator.data_loader import DataError, PriceHistory
from simulator.engine import EXECUTION_MODES, SAME_CLOSE, Simulation
from simulator.storage import save_simulation
from simulator.strategies import STRATEGIES, get_strategy

RUN_NAME_SEPARATOR = "__"


def run_name(strategy_name: str, execution: str = SAME_CLOSE) -> str:
    """The name a run is saved under, e.g. 'spy_buy_and_hold__next_close'.

    Same-close runs keep the plain strategy name, so earlier saved runs still match.
    """
    if execution == SAME_CLOSE:
        return strategy_name
    return f"{strategy_name}{RUN_NAME_SEPARATOR}{execution}"


def run_strategy(
    name: str,
    data_dir: Path = PRICE_DATA_DIR,
    db_path: Path = DATABASE_PATH,
    save: bool = True,
    execution: str = SAME_CLOSE,
) -> Simulation:
    """Replay one registered strategy over every trading day and (by default) save it."""
    spec = get_strategy(name)
    prices = PriceHistory.from_directory(data_dir, spec.tickers)
    sim = Simulation(prices, execution=execution)  # $10,000 cash, $1.00 commission per trade
    sim.run(spec.strategy)
    if save:
        save_simulation(sim, run_name(spec.name, execution), db_path, overwrite=True)
    return sim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run trading strategies and save the results.")
    parser.add_argument("names", nargs="*", metavar="strategy",
                        help="strategy name(s) to run (default: all of them)")
    parser.add_argument("--list", action="store_true", help="list the available strategies")
    parser.add_argument(
        "--execution", choices=EXECUTION_MODES, default=SAME_CLOSE,
        help="same_close: fill at the signal day's close (default); "
             "next_close: fill at the next trading day's close",
    )
    args = parser.parse_args(argv)

    if args.list:
        for spec in STRATEGIES.values():
            print(f"{spec.name}\n    {spec.description}")
        return 0

    try:
        names = args.names or list(STRATEGIES)
        specs = [get_strategy(name) for name in names]  # check every name before running
        for spec in specs:
            sim = run_strategy(spec.name, execution=args.execution)
            last = sim.history[-1]
            print(
                f"{run_name(spec.name, args.execution)}: {len(sim.trades)} trade(s), {last.day}, "
                f"value ${last.total_value:,}, profit/loss ${last.profit_loss:,}  (saved)"
            )
            for rejected in sim.rejected_orders:
                print(f"    rejected {rejected.order.side} {rejected.order.ticker} on {rejected.day}: "
                      f"{rejected.reason}")
            for order in sim.unfilled_orders:
                print(f"    unfilled {order.side} {order.ticker} placed on {order.placed_on} "
                      "(the last day; there is no next day to fill it)")
    except (ValueError, DataError) as error:
        print(f"Could not run strategy: {error}")
        return 1
    print("\nCompare with the S&P 500: python -m simulator.analytics <strategy>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
