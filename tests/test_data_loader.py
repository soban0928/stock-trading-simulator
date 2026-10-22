"""Tests for loading price files.

These use tiny hand-written CSV files (made-up prices, not real market data)
created in a temporary folder, so no internet connection is needed.
"""

from datetime import date
from decimal import Decimal

import pytest

from simulator.data_loader import DataError, PriceHistory, load_price_file


def write_csv(folder, ticker, text):
    path = folder / f"{ticker}.csv"
    path.write_text(text)
    return path


# ----- single file -----------------------------------------------------------

def test_loads_and_sorts_by_date(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n2024-01-03,11.5\n2024-01-02,10.0\n")
    series = load_price_file(path)
    assert list(series.values) == [10.0, 11.5]


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_price_file(tmp_path / "NOPE.csv")


def test_missing_column(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,price\n2024-01-02,10\n")
    with pytest.raises(DataError, match="missing column"):
        load_price_file(path)


def test_empty_file(tmp_path):
    path = write_csv(tmp_path, "AAA", "")
    with pytest.raises(DataError):
        load_price_file(path)


def test_header_only(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n")
    with pytest.raises(DataError, match="no rows"):
        load_price_file(path)


def test_bad_date(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n02/01/2024,10\n")
    with pytest.raises(DataError, match="YYYY-MM-DD"):
        load_price_file(path)


def test_non_numeric_price_reports_line(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n2024-01-02,10\n2024-01-03,abc\n")
    with pytest.raises(DataError, match="line 3"):
        load_price_file(path)


def test_negative_price(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n2024-01-02,-1\n")
    with pytest.raises(DataError, match="positive"):
        load_price_file(path)


def test_duplicate_date(tmp_path):
    path = write_csv(tmp_path, "AAA", "date,close\n2024-01-02,10\n2024-01-02,11\n")
    with pytest.raises(DataError, match="duplicate"):
        load_price_file(path)


# ----- several tickers -------------------------------------------------------

@pytest.fixture
def history(tmp_path):
    write_csv(tmp_path, "AAA", "date,close\n2024-01-02,10.0\n2024-01-03,10.5\n2024-01-04,11.0\n")
    # BBB has no price on 2024-01-03, so that day should be dropped.
    write_csv(tmp_path, "BBB", "date,close\n2024-01-02,50.1234\n2024-01-04,52.0\n")
    return PriceHistory.from_directory(tmp_path, ["AAA", "BBB"])


def test_keeps_only_days_where_every_ticker_has_a_price(history):
    assert history.trading_days == [date(2024, 1, 2), date(2024, 1, 4)]


def test_price_is_exact_decimal(history):
    assert history.price("BBB", date(2024, 1, 2)) == Decimal("50.1234")


def test_prices_on_day(history):
    assert history.prices_on(date(2024, 1, 4)) == {"AAA": Decimal("11.0"), "BBB": Decimal("52.0")}


def test_unknown_ticker_or_day(history):
    with pytest.raises(KeyError):
        history.price("ZZZ", date(2024, 1, 2))
    with pytest.raises(KeyError):
        history.price("AAA", date(2024, 1, 3))  # dropped day


def test_missing_ticker_file_lists_all_missing(tmp_path):
    write_csv(tmp_path, "AAA", "date,close\n2024-01-02,10\n")
    with pytest.raises(DataError, match="BBB, CCC"):
        PriceHistory.from_directory(tmp_path, ["AAA", "BBB", "CCC"])


def test_no_overlapping_days(tmp_path):
    write_csv(tmp_path, "AAA", "date,close\n2024-01-02,10\n")
    write_csv(tmp_path, "BBB", "date,close\n2024-01-03,10\n")
    with pytest.raises(DataError, match="no trading days"):
        PriceHistory.from_directory(tmp_path, ["AAA", "BBB"])
