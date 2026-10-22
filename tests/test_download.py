"""Tests for cleaning downloaded data. No internet needed: the 'download' is
a small fake table shaped like what yfinance returns."""

from datetime import date

import pandas as pd
import pytest

from simulator.download import clean_history


def fake_yfinance_table(closes, dates):
    index = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    return pd.DataFrame({"Open": closes, "Close": closes}, index=index)


def test_keeps_date_and_rounded_close():
    raw = fake_yfinance_table([182.3199920654297], ["2024-01-02"])
    cleaned = clean_history(raw)
    assert list(cleaned.columns) == ["date", "close"]
    assert cleaned["date"].iloc[0] == date(2024, 1, 2)
    assert cleaned["close"].iloc[0] == 182.32


def test_drops_missing_prices_and_sorts():
    raw = fake_yfinance_table([11.0, None, 10.0], ["2024-01-04", "2024-01-03", "2024-01-02"])
    cleaned = clean_history(raw)
    assert list(cleaned["close"]) == [10.0, 11.0]


def test_empty_download_raises():
    with pytest.raises(ValueError):
        clean_history(pd.DataFrame())
