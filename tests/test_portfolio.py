from datetime import date
from decimal import Decimal

import pytest

from simulator.portfolio import (
    InsufficientFundsError,
    InsufficientSharesError,
    InvalidOrderError,
    Portfolio,
)

DAY = date(2024, 1, 2)


@pytest.fixture
def portfolio():
    """A fresh $10,000 portfolio with a $1 commission for each test."""
    return Portfolio(starting_cash="10000", commission="1.00")


# ----- buying ----------------------------------------------------------------

def test_buy_reduces_cash_by_cost_plus_commission(portfolio):
    portfolio.buy("AAPL", 10, "150.00", DAY)
    assert portfolio.cash == Decimal("8499.00")  # 10000 - 1500 - 1
    assert portfolio.quantity_held("AAPL") == Decimal("10")


def test_buy_fractional_shares(portfolio):
    portfolio.buy("MSFT", "0.5", "400.00", DAY)
    assert portfolio.quantity_held("MSFT") == Decimal("0.5")
    assert portfolio.cash == Decimal("9799.00")


def test_buying_twice_updates_average_cost(portfolio):
    portfolio.buy("AAPL", 10, "100", DAY)  # cost 1001
    portfolio.buy("AAPL", 10, "120", DAY)  # cost 1201
    position = portfolio.positions["AAPL"]
    assert position.quantity == Decimal("20")
    assert position.average_cost == Decimal("110.1")  # 2202 / 20


def test_ticker_is_case_insensitive(portfolio):
    portfolio.buy("aapl", 1, "100", DAY)
    assert portfolio.quantity_held("AAPL") == Decimal("1")


def test_cannot_buy_more_than_cash_allows(portfolio):
    with pytest.raises(InsufficientFundsError):
        portfolio.buy("AAPL", 100, "100", DAY)  # needs 10,001
    assert portfolio.cash == Decimal("10000.00")  # nothing changed
    assert portfolio.trades == []


def test_can_spend_exactly_all_cash():
    portfolio = Portfolio(starting_cash="101", commission="1")
    portfolio.buy("AAPL", 1, "100", DAY)
    assert portfolio.cash == Decimal("0.00")


# ----- selling ---------------------------------------------------------------

def test_sell_adds_proceeds_minus_commission(portfolio):
    portfolio.buy("AAPL", 10, "100", DAY)  # cash 8999
    portfolio.sell("AAPL", 4, "110", DAY)  # +440 - 1
    assert portfolio.cash == Decimal("9438.00")
    assert portfolio.quantity_held("AAPL") == Decimal("6")


def test_sell_records_realised_profit(portfolio):
    portfolio.buy("AAPL", 10, "100", DAY)  # total cost 1001 -> 100.10 per share
    trade = portfolio.sell("AAPL", 10, "110", DAY)  # proceeds 1099
    assert trade.realised_pnl == Decimal("98.00")  # 1099 - 1001


def test_partial_sell_keeps_average_cost(portfolio):
    portfolio.buy("AAPL", 10, "100", DAY)
    portfolio.sell("AAPL", 5, "90", DAY)
    assert portfolio.positions["AAPL"].average_cost == Decimal("100.1")


def test_selling_everything_removes_position(portfolio):
    portfolio.buy("AAPL", "0.333333", "300", DAY)
    portfolio.sell("AAPL", "0.333333", "300", DAY)
    assert "AAPL" not in portfolio.positions


def test_cannot_sell_more_than_held(portfolio):
    portfolio.buy("AAPL", 5, "100", DAY)
    with pytest.raises(InsufficientSharesError):
        portfolio.sell("AAPL", 6, "100", DAY)


def test_cannot_sell_stock_not_held(portfolio):
    with pytest.raises(InsufficientSharesError):
        portfolio.sell("TSLA", 1, "200", DAY)


# ----- valuation -------------------------------------------------------------

def test_total_value_uses_current_prices(portfolio):
    portfolio.buy("AAPL", 10, "100", DAY)  # cash 8999
    assert portfolio.total_value({"AAPL": "120"}) == Decimal("10199.00")


def test_market_value_requires_price_for_every_holding(portfolio):
    portfolio.buy("AAPL", 1, "100", DAY)
    with pytest.raises(KeyError):
        portfolio.market_value({})


# ----- input validation ------------------------------------------------------

@pytest.mark.parametrize("bad_quantity", [0, -1, "abc", "NaN", True, "0.0000001"])
def test_rejects_invalid_quantities(portfolio, bad_quantity):
    with pytest.raises(InvalidOrderError):
        portfolio.buy("AAPL", bad_quantity, "100", DAY)


@pytest.mark.parametrize("bad_price", [0, -5, "", None])
def test_rejects_invalid_prices(portfolio, bad_price):
    with pytest.raises(InvalidOrderError):
        portfolio.buy("AAPL", 1, bad_price, DAY)


@pytest.mark.parametrize("bad_ticker", ["", "123", "TOOLONGTICKER", "AA PL", None])
def test_rejects_invalid_tickers(portfolio, bad_ticker):
    with pytest.raises(InvalidOrderError):
        portfolio.buy(bad_ticker, 1, "100", DAY)


def test_rejects_non_date(portfolio):
    with pytest.raises(InvalidOrderError):
        portfolio.buy("AAPL", 1, "100", "2024-01-02")


def test_float_input_is_converted_exactly(portfolio):
    portfolio.buy("AAPL", 0.1, 0.3, DAY)
    assert portfolio.quantity_held("AAPL") == Decimal("0.1")


def test_rejects_bad_starting_settings():
    with pytest.raises(ValueError):
        Portfolio(starting_cash="0")
    with pytest.raises(ValueError):
        Portfolio(commission="-1")
