"""Tests for the caching and context-gathering logic in src/explain.py.

The Anthropic client is mocked throughout so no real API calls happen.
These tests deliberately never assert on the model's actual explanation
text/content (non-deterministic) — only on which flags get sent to the
model, which get skipped as already-explained, and what price history is
gathered for a given ticker/date.
"""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd

from src.explain import get_price_window, get_unexplained_flags, run_explanations
from src.ingest import _TABLE_SCHEMA
from src.validate import _FLAGS_SCHEMA
from src.explain import _EXPLANATIONS_SCHEMA


def _create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_TABLE_SCHEMA)
    con.execute(_FLAGS_SCHEMA)
    con.execute(_EXPLANATIONS_SCHEMA)


def _load(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.register("df_view", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM df_view")
    con.unregister("df_view")


def _weekdays(start: date, count: int) -> list[date]:
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _make_prices(ticker: str, days: list[date], base_price: float, exchange: str = "NYSE") -> pd.DataFrame:
    rows = [
        {
            "ticker": ticker,
            "exchange": exchange,
            "date": d,
            "open": base_price + i,
            "high": base_price + i + 1,
            "low": base_price + i - 1,
            "close": base_price + i,
            "volume": 1000 + i,
            "adj_close": base_price + i,
        }
        for i, d in enumerate(days)
    ]
    return pd.DataFrame(rows)[
        ["ticker", "exchange", "date", "open", "high", "low", "close", "volume", "adj_close"]
    ]


def _mock_anthropic_client(reply_text: str = "mocked explanation") -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=reply_text)],
        stop_reason="end_turn",
    )
    return client


class TestExplain:
    """Caching (SQL join) and context-gathering (price window) logic, with a mocked Anthropic client."""

    def test_get_unexplained_flags_excludes_already_explained(self):
        """A flag with a matching (ticker, date, flag_type) row in `explanations` must not be returned."""
        con = duckdb.connect(":memory:")
        try:
            _create_tables(con)
            days = _weekdays(date(2026, 1, 5), 3)
            flags_df = pd.DataFrame(
                [
                    {"ticker": "AAA", "date": days[0], "flag_type": "MISSING_DATES", "severity": "low", "details": "d1"},
                    {"ticker": "AAA", "date": days[1], "flag_type": "STALE_PRICE", "severity": "medium", "details": "d2"},
                    {"ticker": "BBB", "date": days[2], "flag_type": "DUPLICATE_ROW", "severity": "high", "details": "d3"},
                ]
            )
            _load(con, "flags", flags_df)

            explanations_df = pd.DataFrame(
                [{"ticker": "AAA", "date": days[0], "flag_type": "MISSING_DATES", "explanation": "already explained"}]
            )
            _load(con, "explanations", explanations_df)

            result = get_unexplained_flags(con)
        finally:
            con.close()

        pairs = set(zip(result["ticker"], result["flag_type"]))
        assert len(result) == 2
        assert ("AAA", "MISSING_DATES") not in pairs
        assert ("AAA", "STALE_PRICE") in pairs
        assert ("BBB", "DUPLICATE_ROW") in pairs

    def test_get_unexplained_flags_includes_new_flags_when_explanations_empty(self):
        """With no rows in `explanations` yet, every flag should come back as unexplained."""
        con = duckdb.connect(":memory:")
        try:
            _create_tables(con)
            days = _weekdays(date(2026, 1, 5), 2)
            flags_df = pd.DataFrame(
                [
                    {"ticker": "AAA", "date": days[0], "flag_type": "OHLC_INCONSISTENCY", "severity": "high", "details": "d1"},
                    {"ticker": "AAA", "date": days[1], "flag_type": "OUTLIER_RETURN", "severity": "medium", "details": "d2"},
                ]
            )
            _load(con, "flags", flags_df)

            result = get_unexplained_flags(con)
        finally:
            con.close()

        assert len(result) == 2

    def test_get_price_window_returns_correct_range(self):
        """The window must cover exactly `window` trading days before and after the target date, scoped to the right ticker."""
        con = duckdb.connect(":memory:")
        try:
            _create_tables(con)
            days = _weekdays(date(2026, 1, 5), 30)
            target_idx = 15
            target_date = days[target_idx]

            _load(con, "prices", _make_prices("AAA", days, base_price=10.0))
            _load(con, "prices", _make_prices("BBB", days, base_price=200.0))

            result = get_price_window(con, "AAA", target_date, window=10)
        finally:
            con.close()

        expected_start = days[target_idx - 10]
        expected_end = days[target_idx + 10]

        assert len(result) == 21
        assert result["date"].min() == pd.Timestamp(expected_start)
        assert result["date"].max() == pd.Timestamp(expected_end)
        # Scoped to ticker AAA only (base_price 10.0), never leaks in BBB's rows (base_price 200.0)
        assert (result["close"] < 100).all()

    def test_get_price_window_respects_window_size(self):
        """A smaller window argument should return proportionally fewer rows."""
        con = duckdb.connect(":memory:")
        try:
            _create_tables(con)
            days = _weekdays(date(2026, 1, 5), 30)
            target_idx = 15
            target_date = days[target_idx]

            _load(con, "prices", _make_prices("AAA", days, base_price=10.0))

            result = get_price_window(con, "AAA", target_date, window=3)
        finally:
            con.close()

        assert len(result) == 7  # 3 before + target + 3 after

    def test_run_explanations_skips_cached_and_processes_new(self, tmp_path):
        """run_explanations() must call the (mocked) API only for unexplained flags, and record results."""
        db_path = tmp_path / "market.duckdb"
        config_path = tmp_path / "explain.json"
        config_path.write_text('{"model": "claude-sonnet-5"}', encoding="utf-8")

        days = _weekdays(date(2026, 1, 5), 30)

        con = duckdb.connect(str(db_path))
        try:
            _create_tables(con)
            _load(con, "prices", _make_prices("AAA", days, base_price=10.0))

            flags_df = pd.DataFrame(
                [
                    {"ticker": "AAA", "date": days[10], "flag_type": "STALE_PRICE", "severity": "medium", "details": "d1"},
                    {"ticker": "AAA", "date": days[15], "flag_type": "OUTLIER_RETURN", "severity": "medium", "details": "d2"},
                ]
            )
            _load(con, "flags", flags_df)

            # Pre-seed one explanation so it should be skipped as already cached.
            explanations_df = pd.DataFrame(
                [{"ticker": "AAA", "date": days[10], "flag_type": "STALE_PRICE", "explanation": "pre-existing"}]
            )
            _load(con, "explanations", explanations_df)
        finally:
            con.close()

        mock_client = _mock_anthropic_client()
        with patch("src.explain.anthropic.Anthropic", return_value=mock_client):
            summary = run_explanations(db_path=db_path, config_path=config_path)

        assert summary["already_cached"] == 1
        assert summary["newly_explained"] == 1
        assert summary["failed"] == 0
        assert mock_client.messages.create.call_count == 1

        con = duckdb.connect(str(db_path))
        try:
            total_explanations = con.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]
            new_row = con.execute(
                "SELECT ticker, date, flag_type FROM explanations WHERE flag_type = 'OUTLIER_RETURN'"
            ).fetchone()
        finally:
            con.close()

        assert total_explanations == 2
        assert new_row == ("AAA", pd.Timestamp(days[15]).to_pydatetime().date(), "OUTLIER_RETURN")

    def test_run_explanations_logs_and_continues_on_api_failure(self, tmp_path):
        """A raised exception from the (mocked) API for one flag must not crash the whole batch."""
        db_path = tmp_path / "market.duckdb"
        config_path = tmp_path / "explain.json"
        config_path.write_text('{"model": "claude-sonnet-5"}', encoding="utf-8")

        days = _weekdays(date(2026, 1, 5), 30)

        con = duckdb.connect(str(db_path))
        try:
            _create_tables(con)
            _load(con, "prices", _make_prices("AAA", days, base_price=10.0))
            flags_df = pd.DataFrame(
                [
                    {"ticker": "AAA", "date": days[10], "flag_type": "STALE_PRICE", "severity": "medium", "details": "d1"},
                    {"ticker": "AAA", "date": days[15], "flag_type": "OUTLIER_RETURN", "severity": "medium", "details": "d2"},
                ]
            )
            _load(con, "flags", flags_df)
        finally:
            con.close()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("simulated API failure")

        with patch("src.explain.anthropic.Anthropic", return_value=mock_client):
            summary = run_explanations(db_path=db_path, config_path=config_path)

        assert summary["failed"] == 2
        assert summary["newly_explained"] == 0

        con = duckdb.connect(str(db_path))
        try:
            total_explanations = con.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]
        finally:
            con.close()

        assert total_explanations == 0
