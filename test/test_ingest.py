"""Tests for the yfinance-to-DuckDB pipeline in src/ingest.py.

`yf.Ticker(...).history(...)` is mocked via unittest.mock so no real network
requests happen. Each test writes to a temp-file DuckDB database (never
data/market_data.duckdb) using pytest's `tmp_path` fixture.
"""

from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from src import ingest


def _make_history(dates, opens, highs, lows, closes, volumes, adj_closes) -> pd.DataFrame:
    """Build a DataFrame shaped like yfinance's Ticker.history() output."""
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
            "Adj Close": adj_closes,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _patch_ticker(history_by_ticker: dict):
    """Return a patch context that makes yf.Ticker(ticker).history(...) return
    the DataFrame configured for that ticker in `history_by_ticker`."""

    def _side_effect(ticker, *args, **kwargs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = history_by_ticker[ticker]
        return mock_ticker

    return patch("src.ingest.yf.Ticker", side_effect=_side_effect)


class TestIngest:
    """Verifies fetch/normalize/write behavior against mocked yfinance data."""

    def test_ingest_normal_data(self, tmp_path):
        """Well-formed multi-ticker data should land in DuckDB with correct row count, columns, and values."""
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        aaa_hist = _make_history(
            dates,
            opens=[10.0, 11.0, 12.0, 13.0, 14.0],
            highs=[10.5, 11.5, 12.5, 13.5, 14.5],
            lows=[9.5, 10.5, 11.5, 12.5, 13.5],
            closes=[10.2, 11.2, 12.2, 13.2, 14.2],
            volumes=[1000, 1100, 1200, 1300, 1400],
            adj_closes=[10.1, 11.1, 12.1, 13.1, 14.1],
        )
        bbb_hist = _make_history(
            dates,
            opens=[50.0, 51.0, 52.0, 53.0, 54.0],
            highs=[50.5, 51.5, 52.5, 53.5, 54.5],
            lows=[49.5, 50.5, 51.5, 52.5, 53.5],
            closes=[50.2, 51.2, 52.2, 53.2, 54.2],
            volumes=[2000, 2100, 2200, 2300, 2400],
            adj_closes=[50.1, 51.1, 52.1, 53.1, 54.1],
        )

        with _patch_ticker({"AAA": aaa_hist, "BBB": bbb_hist}):
            df = ingest.fetch_all(["AAA", "BBB"], period="5d")

        db_path = tmp_path / "market.duckdb"
        ingest.write_to_duckdb(df, db_path)

        con = duckdb.connect(str(db_path))
        try:
            result = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()
        finally:
            con.close()

        assert len(result) == 10
        assert list(result.columns) == [
            "ticker", "date", "open", "high", "low", "close", "volume", "adj_close",
        ]

        aaa_first = result[result["ticker"] == "AAA"].iloc[0]
        assert aaa_first["open"] == pytest.approx(10.0)
        assert aaa_first["close"] == pytest.approx(10.2)
        assert aaa_first["volume"] == 1000

        bbb_last = result[result["ticker"] == "BBB"].iloc[-1]
        assert bbb_last["adj_close"] == pytest.approx(54.1)

    def test_ingest_handles_missing_values(self, tmp_path):
        """NaNs in open/close should be written through as-is, not dropped or crash ingest.py."""
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        aaa_hist = _make_history(
            dates,
            opens=[10.0, None, 12.0, 13.0, 14.0],
            highs=[10.5, 11.5, 12.5, 13.5, 14.5],
            lows=[9.5, 10.5, 11.5, 12.5, 13.5],
            closes=[10.2, 11.2, 12.2, None, 14.2],
            volumes=[1000, 1100, 1200, 1300, 1400],
            adj_closes=[10.1, 11.1, 12.1, 13.1, 14.1],
        )

        with _patch_ticker({"AAA": aaa_hist}):
            df = ingest.fetch_all(["AAA"], period="5d")

        db_path = tmp_path / "market.duckdb"
        ingest.write_to_duckdb(df, db_path)

        con = duckdb.connect(str(db_path))
        try:
            result = con.execute("SELECT * FROM prices ORDER BY date").fetchdf()
        finally:
            con.close()

        assert len(result) == 5
        assert pd.isna(result.iloc[1]["open"])
        assert pd.isna(result.iloc[3]["close"])
        # rows around the NaNs should be untouched
        assert result.iloc[0]["open"] == pytest.approx(10.0)
        assert result.iloc[4]["close"] == pytest.approx(14.2)

    def test_ingest_handles_empty_response(self, monkeypatch):
        """A ticker that only ever returns an empty DataFrame should raise a clear, specific exception."""
        monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

        empty_hist = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
            index=pd.DatetimeIndex([], name="Date"),
        )

        with _patch_ticker({"XXX": empty_hist}):
            with pytest.raises(ingest.TickerDataUnavailable, match="XXX"):
                ingest.fetch_all(["XXX"], period="5d")

    def test_ingest_rerun_does_not_duplicate(self, tmp_path):
        """Running ingest twice with the same data/date range must not duplicate (ticker, date) rows."""
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        aaa_hist = _make_history(
            dates,
            opens=[10.0, 11.0, 12.0, 13.0, 14.0],
            highs=[10.5, 11.5, 12.5, 13.5, 14.5],
            lows=[9.5, 10.5, 11.5, 12.5, 13.5],
            closes=[10.2, 11.2, 12.2, 13.2, 14.2],
            volumes=[1000, 1100, 1200, 1300, 1400],
            adj_closes=[10.1, 11.1, 12.1, 13.1, 14.1],
        )
        bbb_hist = _make_history(
            dates,
            opens=[50.0, 51.0, 52.0, 53.0, 54.0],
            highs=[50.5, 51.5, 52.5, 53.5, 54.5],
            lows=[49.5, 50.5, 51.5, 52.5, 53.5],
            closes=[50.2, 51.2, 52.2, 53.2, 54.2],
            volumes=[2000, 2100, 2200, 2300, 2400],
            adj_closes=[50.1, 51.1, 52.1, 53.1, 54.1],
        )

        db_path = tmp_path / "market.duckdb"

        with _patch_ticker({"AAA": aaa_hist, "BBB": bbb_hist}):
            ingest.write_to_duckdb(ingest.fetch_all(["AAA", "BBB"], period="5d"), db_path)
            ingest.write_to_duckdb(ingest.fetch_all(["AAA", "BBB"], period="5d"), db_path)

        con = duckdb.connect(str(db_path))
        try:
            total = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            distinct = con.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT ticker, date FROM prices)"
            ).fetchone()[0]
        finally:
            con.close()

        assert total == 10
        assert total == distinct
