"""Core portfolio engine: cash, positions, buying and selling.

This module knows nothing about dates, price files or the user interface.
It only enforces the rules of trading. Keeping it independent makes it
easy to test on its own.

Money and share quantities use Decimal rather than float, because floats
cannot represent most decimal values exactly (0.1 + 0.2 != 0.3 in Python).
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
import re

CENT = Decimal("0.01")
MAX_QUANTITY_DECIMALS = 6  # fractional shares down to 0.000001
SHARE_STEP = Decimal(1).scaleb(-MAX_QUANTITY_DECIMALS)  # 0.000001
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")  # e.g. AAPL, BRK-B


class TradingError(Exception):
    """Base class for every error a trade can raise."""


class InvalidOrderError(TradingError):
    """The order itself is malformed (bad ticker, negative quantity...)."""


class InsufficientFundsError(TradingError):
    """Not enough cash to pay for a trade."""


class InsufficientSharesError(TradingError):
    """Trying to sell more shares than are held."""


def to_money(value) -> Decimal:
    """Round a Decimal to whole cents (half-up, as people expect)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def to_decimal(value, field_name: str) -> Decimal:
    """Convert user input (int, str, Decimal or float) into a Decimal.

    Floats go through str() first so 0.1 becomes Decimal("0.1") rather
    than Decimal("0.1000000000000000055511151231257827...").
    """
    if isinstance(value, bool):  # bool is a subclass of int; reject it
        raise InvalidOrderError(f"{field_name} must be a number, not a boolean")
    if isinstance(value, float):
        value = str(value)
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise InvalidOrderError(f"{field_name} must be a number, got {value!r}")
    if not result.is_finite():
        raise InvalidOrderError(f"{field_name} must be a finite number")
    return result


def affordable_quantity(cash: Decimal, commission: Decimal, price: Decimal) -> Decimal:
    """The most shares that cash can buy at price after paying one commission.

    Rounded *down* to the smallest share step, so the trade never costs more
    than the cash available. Returns 0 if the commission alone uses up the cash.
    """
    budget = cash - commission
    if budget <= 0:
        return Decimal("0")
    return (budget / price).quantize(SHARE_STEP, rounding=ROUND_DOWN)


def normalise_ticker(ticker) -> str:
    """Upper-case and validate a ticker symbol."""
    if not isinstance(ticker, str):
        raise InvalidOrderError("ticker must be a string")
    cleaned = ticker.strip().upper()
    if not TICKER_PATTERN.match(cleaned):
        raise InvalidOrderError(f"invalid ticker symbol: {ticker!r}")
    return cleaned


@dataclass(frozen=True)
class Trade:
    """An executed trade. frozen=True means it cannot be changed later."""

    trade_date: date
    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    commission: Decimal
    gross_value: Decimal  # quantity * price, rounded to cents
    realised_pnl: Decimal  # profit/loss locked in by a SELL; zero for BUY


@dataclass
class Position:
    """Shares held in one stock, plus what they cost in total.

    total_cost includes the commission paid when buying, so the average
    cost reflects what you really paid per share.
    """

    quantity: Decimal
    total_cost: Decimal

    @property
    def average_cost(self) -> Decimal:
        return self.total_cost / self.quantity


class Portfolio:
    """A cash balance plus a set of stock positions."""

    def __init__(self, starting_cash=Decimal("10000"), commission=Decimal("1.00")):
        starting_cash = to_decimal(starting_cash, "starting_cash")
        commission = to_decimal(commission, "commission")
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if commission < 0:
            raise ValueError("commission cannot be negative")

        self.starting_cash = to_money(starting_cash)
        self.commission = to_money(commission)
        self.cash = self.starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    # ----- queries -------------------------------------------------------

    def quantity_held(self, ticker: str) -> Decimal:
        position = self.positions.get(normalise_ticker(ticker))
        return position.quantity if position else Decimal("0")

    def market_value(self, prices: dict[str, Decimal]) -> Decimal:
        """Value of all shares at the given prices (ticker -> price)."""
        total = Decimal("0")
        for ticker, position in self.positions.items():
            if ticker not in prices:
                raise KeyError(f"no price provided for held stock {ticker}")
            total += position.quantity * to_decimal(prices[ticker], "price")
        return to_money(total)

    def total_value(self, prices: dict[str, Decimal]) -> Decimal:
        return self.cash + self.market_value(prices)

    # ----- trading -------------------------------------------------------

    def buy(self, ticker, quantity, price, trade_date: date) -> Trade:
        ticker, quantity, price = self._validate_order(ticker, quantity, price, trade_date)
        gross_value = to_money(quantity * price)
        total_cost = gross_value + self.commission

        if total_cost > self.cash:
            raise InsufficientFundsError(
                f"buying {quantity} {ticker} costs ${total_cost} "
                f"(including ${self.commission} commission) but cash is ${self.cash}"
            )

        self.cash -= total_cost
        position = self.positions.get(ticker)
        if position is None:
            self.positions[ticker] = Position(quantity, total_cost)
        else:
            position.quantity += quantity
            position.total_cost += total_cost

        return self._record(trade_date, ticker, "BUY", quantity, price, gross_value, Decimal("0"))

    def sell(self, ticker, quantity, price, trade_date: date) -> Trade:
        ticker, quantity, price = self._validate_order(ticker, quantity, price, trade_date)
        held = self.quantity_held(ticker)
        if quantity > held:
            raise InsufficientSharesError(f"cannot sell {quantity} {ticker}: only {held} held")

        gross_value = to_money(quantity * price)
        proceeds = gross_value - self.commission
        if self.cash + proceeds < 0:
            # Only possible for tiny sales where commission exceeds the sale value.
            raise InsufficientFundsError("not enough cash to pay the commission on this sale")

        position = self.positions[ticker]
        if quantity == position.quantity:
            # Selling everything: remove the whole cost so no stray pennies remain.
            cost_of_sold_shares = position.total_cost
            del self.positions[ticker]
        else:
            # Average-cost method: each share carries the same share of total cost.
            cost_of_sold_shares = to_money(position.total_cost * quantity / position.quantity)
            position.quantity -= quantity
            position.total_cost -= cost_of_sold_shares

        self.cash += proceeds
        realised_pnl = proceeds - cost_of_sold_shares
        return self._record(trade_date, ticker, "SELL", quantity, price, gross_value, realised_pnl)

    # ----- helpers -------------------------------------------------------

    def _validate_order(self, ticker, quantity, price, trade_date):
        ticker = normalise_ticker(ticker)
        quantity = to_decimal(quantity, "quantity")
        price = to_decimal(price, "price")

        if quantity <= 0:
            raise InvalidOrderError("quantity must be greater than zero")
        if -quantity.as_tuple().exponent > MAX_QUANTITY_DECIMALS:
            raise InvalidOrderError(
                f"quantity can have at most {MAX_QUANTITY_DECIMALS} decimal places"
            )
        if price <= 0:
            raise InvalidOrderError("price must be greater than zero")
        if not isinstance(trade_date, date):
            raise InvalidOrderError("trade_date must be a datetime.date")
        return ticker, quantity, price

    def _record(self, trade_date, ticker, side, quantity, price, gross_value, realised_pnl):
        trade = Trade(
            trade_date=trade_date,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            commission=self.commission,
            gross_value=gross_value,
            realised_pnl=realised_pnl,
        )
        self.trades.append(trade)
        return trade
