"""Tests for the SQL-based data quality checks in src/validate.py.

Each check gets a "clean" case (no issues, zero flags expected) and a
"dirty" case (exactly one instance of that issue introduced, expecting
exactly the matching row(s) flagged with the right flag_type/severity).
All fixtures are small, hardcoded DataFrames loaded into an in-memory
DuckDB connection (":memory:") so tests never call yfinance or touch
data/market_data.duckdb.
"""

from datetime import date, timedelta
from itertools import cycle

import duckdb
import pandas as pd
import pandas_market_calendars as mcal

from src.validate import (
    check_duplicate_rows,
    check_missing_dates,
    check_ohlc_inconsistency,
    check_outlier_return,
    check_stale_price,
)

_COLUMNS = ["ticker", "exchange", "date", "open", "high", "low", "close", "volume", "adj_close"]


def _make_prices_df(rows: list[dict], exchange: str = "NYSE") -> pd.DataFrame:
    """Build a prices-shaped DataFrame from a list of row dicts. Rows without an
    explicit 'exchange' key default to `exchange`."""
    df = pd.DataFrame(rows)
    if "exchange" not in df.columns:
        df["exchange"] = exchange
    return df[_COLUMNS]


def _run_check(check_fn, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Load `df` into an in-memory `prices` table and run the given check function against it."""
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE prices (
                ticker VARCHAR, exchange VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
                close DOUBLE, volume BIGINT, adj_close DOUBLE
            )
            """
        )
        con.register("df_view", df)
        con.execute("INSERT INTO prices SELECT * FROM df_view")
        return check_fn(con, **kwargs)
    finally:
        con.close()


def _weekdays(start: date, count: int) -> list[date]:
    """Return `count` sequential weekdays (Mon-Fri), starting from `start` and skipping weekends."""
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _trading_days(exchange: str, start: date, end: date) -> list[date]:
    """Return actual trading days for `exchange` between start and end, via pandas_market_calendars."""
    calendar = mcal.get_calendar(exchange)
    schedule = calendar.valid_days(start_date=start, end_date=end)
    return [ts.date() for ts in schedule]


def _prices_from_returns(start_price: float, pct_returns: list[float]) -> list[float]:
    """Compound a list of percentage returns onto a starting price."""
    prices = []
    price = start_price
    for pct in pct_returns:
        price = price * (1 + pct / 100)
        prices.append(round(price, 6))
    return prices


class TestValidateChecks:
    """One clean/dirty pair per check in validate.py, so a failure pinpoints the broken check."""

    # ---------- MISSING_DATES ----------

    def test_missing_dates_clean(self):
        """A run of consecutive real NYSE trading days with no gaps should produce zero MISSING_DATES flags."""
        days = _trading_days("NYSE", date(2026, 1, 5), date(2026, 2, 15))[:8]
        rows = [
            {"ticker": "AAA", "exchange": "NYSE", "date": d, "open": 10, "high": 11, "low": 9,
             "close": 10 + i, "volume": 1000, "adj_close": 10 + i}
            for i, d in enumerate(days)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_missing_dates, df)

        assert result.empty

    def test_missing_dates_dirty(self):
        """Removing one real trading day from the middle of the range should flag exactly that date."""
        days = _trading_days("NYSE", date(2026, 1, 5), date(2026, 2, 15))[:8]
        missing_day = days[3]
        remaining_days = [d for d in days if d != missing_day]
        rows = [
            {"ticker": "AAA", "exchange": "NYSE", "date": d, "open": 10, "high": 11, "low": 9,
             "close": 10 + i, "volume": 1000, "adj_close": 10 + i}
            for i, d in enumerate(remaining_days)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_missing_dates, df)

        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAA"
        assert result.iloc[0]["date"] == pd.Timestamp(missing_day)
        assert result.iloc[0]["flag_type"] == "MISSING_DATES"
        assert result.iloc[0]["severity"] == "low"

    def test_missing_dates_never_flags_a_holiday(self):
        """A US market holiday within the range (no row exists for it) must never be flagged as missing."""
        # 2026-01-19 is Martin Luther King Jr. Day (NYSE closed) — deliberately excluded from `days`,
        # but also must not appear in the check's own expected-trading-day set.
        days = _trading_days("NYSE", date(2026, 1, 5), date(2026, 1, 23))
        assert date(2026, 1, 19) not in days  # sanity-check our own assumption about the holiday

        rows = [
            {"ticker": "AAA", "exchange": "NYSE", "date": d, "open": 10, "high": 11, "low": 9,
             "close": 10 + i, "volume": 1000, "adj_close": 10 + i}
            for i, d in enumerate(days)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_missing_dates, df)

        assert result.empty

    # ---------- STALE_PRICE ----------

    def test_stale_price_clean(self):
        """Closes that change every day should produce zero STALE_PRICE flags (default n=3)."""
        days = _weekdays(date(2026, 1, 5), 6)
        closes = [10, 11, 12, 13, 14, 15]
        rows = [
            {"ticker": "AAA", "date": d, "open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 1000, "adj_close": c}
            for d, c in zip(days, closes)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_stale_price, df, n=3)

        assert result.empty

    def test_stale_price_dirty(self):
        """Repeating the same close for 4 consecutive days should flag the day the streak passes n=3."""
        days = _weekdays(date(2026, 1, 5), 6)
        closes = [10, 11, 12, 12, 12, 12]  # close=12 repeated on days[2..5]
        rows = [
            {"ticker": "AAA", "date": d, "open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 1000, "adj_close": c}
            for d, c in zip(days, closes)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_stale_price, df, n=3)

        assert len(result) == 1
        assert result.iloc[0]["date"] == pd.Timestamp(days[5])
        assert result.iloc[0]["flag_type"] == "STALE_PRICE"
        assert result.iloc[0]["severity"] == "medium"
        assert "4 consecutive trading days" in result.iloc[0]["details"]

    # ---------- OUTLIER_RETURN ----------

    def test_outlier_return_clean(self):
        """Small, varied daily returns with no anomaly should produce zero OUTLIER_RETURN flags."""
        pattern = cycle([0.5, -0.3, 0.4, -0.2, 0.6, -0.4, 0.3, -0.5, 0.2, -0.1])
        days = _weekdays(date(2026, 1, 5), 33)
        returns = [next(pattern) for _ in range(32)]
        closes = [100.0] + _prices_from_returns(100.0, returns)
        rows = [
            {"ticker": "AAA", "date": d, "open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 1000, "adj_close": c}
            for d, c in zip(days, closes)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_outlier_return, df, k=3.0)

        assert result.empty

    def test_outlier_return_dirty(self):
        """A single huge one-day jump after weeks of steady returns should be flagged as OUTLIER_RETURN."""
        pattern = cycle([0.5, -0.3, 0.4, -0.2, 0.6, -0.4, 0.3, -0.5, 0.2, -0.1])
        days = _weekdays(date(2026, 1, 5), 33)
        returns = [next(pattern) for _ in range(31)] + [25.0]  # final day: +25% jump
        closes = [100.0] + _prices_from_returns(100.0, returns)
        rows = [
            {"ticker": "AAA", "date": d, "open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 1000, "adj_close": c}
            for d, c in zip(days, closes)
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_outlier_return, df, k=3.0)

        assert len(result) == 1
        assert result.iloc[0]["date"] == pd.Timestamp(days[-1])
        assert result.iloc[0]["flag_type"] == "OUTLIER_RETURN"
        assert result.iloc[0]["severity"] == "medium"

    # ---------- DUPLICATE_ROW ----------

    def test_duplicate_row_clean(self):
        """Unique (ticker, date) pairs should produce zero DUPLICATE_ROW flags."""
        days = _weekdays(date(2026, 1, 5), 5)
        rows = [
            {"ticker": "AAA", "date": d, "open": 10, "high": 11, "low": 9, "close": 10,
             "volume": 1000, "adj_close": 10}
            for d in days
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_duplicate_rows, df)

        assert result.empty

    def test_duplicate_row_dirty(self):
        """Duplicating a single row's (ticker, date) should flag exactly that combination."""
        days = _weekdays(date(2026, 1, 5), 5)
        rows = [
            {"ticker": "AAA", "date": d, "open": 10, "high": 11, "low": 9, "close": 10,
             "volume": 1000, "adj_close": 10}
            for d in days
        ]
        rows.append(dict(rows[2]))  # duplicate the row for days[2]
        df = _make_prices_df(rows)

        result = _run_check(check_duplicate_rows, df)

        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAA"
        assert result.iloc[0]["date"] == pd.Timestamp(days[2])
        assert result.iloc[0]["flag_type"] == "DUPLICATE_ROW"
        assert result.iloc[0]["severity"] == "high"

    # ---------- OHLC_INCONSISTENCY ----------

    def test_ohlc_inconsistency_clean(self):
        """Internally consistent OHLC rows (high >= all, low <= all) should produce zero flags."""
        days = _weekdays(date(2026, 1, 5), 5)
        rows = [
            {"ticker": "AAA", "date": d, "open": 10, "high": 12, "low": 8, "close": 11,
             "volume": 1000, "adj_close": 11}
            for d in days
        ]
        df = _make_prices_df(rows)

        result = _run_check(check_ohlc_inconsistency, df)

        assert result.empty

    def test_ohlc_inconsistency_dirty(self):
        """A row where high < low should be flagged as OHLC_INCONSISTENCY."""
        days = _weekdays(date(2026, 1, 5), 5)
        rows = [
            {"ticker": "AAA", "date": d, "open": 10, "high": 12, "low": 8, "close": 11,
             "volume": 1000, "adj_close": 11}
            for d in days
        ]
        rows[2]["high"] = 5  # high (5) < low (8) on days[2]
        df = _make_prices_df(rows)

        result = _run_check(check_ohlc_inconsistency, df)

        assert len(result) == 1
        assert result.iloc[0]["date"] == pd.Timestamp(days[2])
        assert result.iloc[0]["flag_type"] == "OHLC_INCONSISTENCY"
        assert result.iloc[0]["severity"] == "high"
