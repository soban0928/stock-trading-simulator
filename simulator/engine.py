"""Day-by-day trading engine: replays historical prices one trading day at a time.

The engine keeps a "current day". You can only see prices up to that day and
can only trade at that day's closing price. Days only move forward, so a
decision can never use a price from the future. (Using future prices by
mistake is called look-ahead bias, and it makes a strategy look far better
than it would have been in real life.)

The trading rules themselves (cash, commission, fractional shares, not
overspending or overselling) live in simulator.portfolio; this module only
decides *when* and *at what price* trades happen.

Execution settings (when an order is filled):
    "same_close" (default)  an order on day t fills at day t's close, straight away.
    "next_close"            an order on day t waits, and fills at the close of the
                            next trading day (t+1). An order placed on the last
                            day is never filled.
The price files only contain closing prices, so there is no "next open" option.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable

import pandas as pd

from simulator.data_loader import PriceHistory
from simulator.portfolio import (
    InvalidOrderError,
    Portfolio,
    Trade,
    TradingError,
    affordable_quantity,
    normalise_ticker,
    to_decimal,
)

SAME_CLOSE = "same_close"
NEXT_CLOSE = "next_close"
EXECUTION_MODES = (SAME_CLOSE, NEXT_CLOSE)


class SimulationFinishedError(Exception):
    """Tried to trade or move forward after the last trading day."""


class ExecutionError(Exception):
    """Tried to trade in a way the execution setting doesn't allow."""


@dataclass(frozen=True)
class Order:
    """A request to buy or sell, waiting to be filled."""

    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: Decimal | None  # None = all the cash (BUY) or all the shares (SELL)
    placed_on: date


@dataclass(frozen=True)
class RejectedOrder:
    """An order that could not be filled on its fill day, and why."""

    order: Order
    day: date
    reason: str


@dataclass(frozen=True)
class DailySnapshot:
    """The portfolio at the close of one trading day, after that day's trades."""

    day: date
    cash: Decimal
    market_value: Decimal  # shares held, valued at the day's closing prices
    total_value: Decimal  # cash + market_value
    profit_loss: Decimal  # total_value minus the starting cash
    holdings: dict[str, Decimal]  # ticker -> shares held


class Simulation:
    """Steps through trading days in date order, trading at each day's close.

    Typical use, one day at a time:
        sim = Simulation(prices)
        sim.buy("AAPL", 2)      # at today's closing price
        sim.end_day()           # record today, move to the next day

    Or hand the whole replay to a strategy function with sim.run(strategy).

    With execution="next_close", use submit_order() instead of buy()/sell():
        sim.submit_order("AAPL", "BUY", 2)   # waits...
        sim.end_day()                        # ...and fills at the next day's close
    """

    def __init__(
        self,
        prices: PriceHistory,
        portfolio: Portfolio | None = None,
        start: date | None = None,
        end: date | None = None,
        execution: str = SAME_CLOSE,
    ):
        if execution not in EXECUTION_MODES:
            raise ValueError(f"execution must be one of {EXECUTION_MODES}, got {execution!r}")
        self.execution = execution
        self.pending_orders: list[Order] = []  # waiting for the next trading day
        self.rejected_orders: list[RejectedOrder] = []
        self._prices = prices
        # A default Portfolio starts with $10,000 and a $1.00 commission.
        self.portfolio = portfolio if portfolio is not None else Portfolio()
        self._days = [
            day
            for day in prices.trading_days
            if (start is None or day >= start) and (end is None or day <= end)
        ]
        if not self._days:
            raise ValueError("no trading days between start and end")
        self._index = 0  # position of the current day in self._days
        self.history: list[DailySnapshot] = []

    # ----- where we are --------------------------------------------------

    @property
    def finished(self) -> bool:
        return self._index >= len(self._days)

    @property
    def current_day(self) -> date:
        self._require_running()
        return self._days[self._index]

    @property
    def trades(self) -> list[Trade]:
        return self.portfolio.trades

    # ----- what you are allowed to see -----------------------------------

    def price(self, ticker: str) -> Decimal:
        """Today's closing price of one ticker."""
        return self._prices.price(normalise_ticker(ticker), self.current_day)

    def prices_today(self) -> dict[str, Decimal]:
        return self._prices.prices_on(self.current_day)

    def past_closes(self, ticker: str) -> pd.Series:
        """Closing prices from the first day of the data up to and including today."""
        return self._prices.closes_through(normalise_ticker(ticker), self.current_day)

    # ----- trading -------------------------------------------------------

    def buy(self, ticker: str, quantity) -> Trade:
        """Buy at today's closing price. Raises a TradingError if not allowed."""
        self._require_same_close("buy")
        return self.portfolio.buy(ticker, quantity, self.price(ticker), self.current_day)

    def sell(self, ticker: str, quantity) -> Trade:
        """Sell at today's closing price. Raises a TradingError if not allowed."""
        self._require_same_close("sell")
        return self.portfolio.sell(ticker, quantity, self.price(ticker), self.current_day)

    def submit_order(self, ticker: str, side: str, quantity=None) -> Trade | None:
        """Place an order; it is filled according to the execution setting.

        quantity=None means "all the cash" for a BUY (after commission, rounded
        down to the smallest fractional share) or "all the shares" for a SELL.
        That amount is worked out on the day the order is filled.

        "same_close": fills now and returns the Trade (or None if there was
        nothing to buy or sell). Raises a TradingError if a fixed quantity
        isn't allowed, just like buy() and sell().
        "next_close": queues the order and returns None. If it can't be filled
        on the next day (for example, not enough cash), it is added to
        rejected_orders instead of stopping the simulation.
        """
        ticker = normalise_ticker(ticker)
        if side not in ("BUY", "SELL"):
            raise InvalidOrderError(f"side must be 'BUY' or 'SELL', got {side!r}")
        if quantity is not None:
            quantity = to_decimal(quantity, "quantity")
            if quantity <= 0:
                raise InvalidOrderError("quantity must be greater than zero")
        self.price(ticker)  # fails now, not tomorrow, if the ticker has no prices

        order = Order(ticker, side, quantity, placed_on=self.current_day)
        if self.execution == SAME_CLOSE:
            return self._fill(order)
        self.pending_orders.append(order)
        return None

    def _fill(self, order: Order) -> Trade | None:
        """Execute an order at the current day's close."""
        price = self.price(order.ticker)
        if order.side == "BUY":
            quantity = order.quantity
            if quantity is None:
                quantity = affordable_quantity(self.portfolio.cash, self.portfolio.commission, price)
            if quantity <= 0:
                return None
            return self.portfolio.buy(order.ticker, quantity, price, self.current_day)
        quantity = order.quantity if order.quantity is not None else self.portfolio.quantity_held(order.ticker)
        if quantity <= 0:
            return None
        return self.portfolio.sell(order.ticker, quantity, price, self.current_day)

    def _fill_pending_orders(self):
        """Fill yesterday's orders at today's close: sells first, so their cash can fund buys."""
        orders, self.pending_orders = self.pending_orders, []
        for order in sorted(orders, key=lambda o: o.side != "SELL"):  # stable: keeps order otherwise
            try:
                self._fill(order)
            except TradingError as error:
                self.rejected_orders.append(RejectedOrder(order, self.current_day, str(error)))

    @property
    def unfilled_orders(self) -> list[Order]:
        """Orders still waiting when the simulation ended (placed on the last day)."""
        return list(self.pending_orders) if self.finished else []

    # ----- moving through time -------------------------------------------

    def end_day(self) -> DailySnapshot:
        """Record the portfolio at today's close, then move to the next trading day.

        Orders waiting from today are filled at the next day's close, as soon as
        the simulation arrives there and before the strategy runs for that day.
        """
        prices = self.prices_today()
        market_value = self.portfolio.market_value(prices)
        total_value = self.portfolio.cash + market_value
        snapshot = DailySnapshot(
            day=self.current_day,
            cash=self.portfolio.cash,
            market_value=market_value,
            total_value=total_value,
            profit_loss=total_value - self.portfolio.starting_cash,
            holdings={t: p.quantity for t, p in self.portfolio.positions.items()},
        )
        self.history.append(snapshot)
        self._index += 1
        if self.pending_orders and not self.finished:
            self._fill_pending_orders()
        return snapshot

    def run(self, strategy: Callable[["Simulation"], None]) -> list[DailySnapshot]:
        """Call strategy(sim) once for each remaining trading day, oldest first.

        The strategy decides what to buy or sell using what the simulation lets
        it see (price, past_closes, portfolio). After it returns, the day is
        recorded and the simulation moves on.
        """
        while not self.finished:
            strategy(self)
            self.end_day()
        return self.history

    def _require_same_close(self, action: str):
        if self.execution != SAME_CLOSE:
            raise ExecutionError(
                f"{action}() trades at today's close, which {self.execution!r} execution "
                "doesn't allow; use submit_order() so the trade fills on the next day"
            )

    def _require_running(self):
        if self.finished:
            raise SimulationFinishedError(f"simulation ended on {self._days[-1]}")
