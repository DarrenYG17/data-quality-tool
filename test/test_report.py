"""Tests for the markdown report assembled by src/report.py.

Builds small DuckDB fixtures (prices/flags/explanations) in a temp-file
database via pytest's tmp_path fixture — never data/market_data.duckdb — and
checks the assembled markdown for the specific facts it should contain, not
exact string equality (formatting details could reasonably change).
"""

from datetime import date, timedelta

import duckdb
import pandas as pd

from src.explain import _EXPLANATIONS_SCHEMA
from src.ingest import _TABLE_SCHEMA
from src.report import build_report, default_report_filename, save_report
from src.validate import _FLAGS_SCHEMA


def _weekdays(start: date, count: int) -> list[date]:
    """Return `count` sequential weekdays (Mon-Fri), starting from `start` and skipping weekends."""
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _make_price_row(ticker: str, exchange: str, d: date, price: float) -> dict:
    return {
        "ticker": ticker, "exchange": exchange, "date": d,
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "volume": 1000, "adj_close": price,
    }


def _setup_db(
    db_path,
    prices_rows: list[dict],
    flags_rows: list[dict],
    explanations_rows: list[dict] | None = None,
) -> None:
    """Create the standard prices/flags/explanations tables in a fresh DuckDB file and load rows."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_TABLE_SCHEMA)
        con.execute(_FLAGS_SCHEMA)
        con.execute(_EXPLANATIONS_SCHEMA)

        if prices_rows:
            df = pd.DataFrame(prices_rows)
            con.register("p", df)
            con.execute("INSERT INTO prices SELECT * FROM p")

        if flags_rows:
            df = pd.DataFrame(flags_rows)
            con.register("f", df)
            con.execute("INSERT INTO flags SELECT * FROM f")

        if explanations_rows:
            df = pd.DataFrame(explanations_rows)
            con.register("e", df)
            con.execute("INSERT INTO explanations SELECT * FROM e")
    finally:
        con.close()


class TestReport:
    """Checks the assembled markdown contains the right facts, not exact string equality."""

    def test_summary_section_reports_range_tickers_and_counts(self, tmp_path):
        """Summary should show the correct date range, ticker list, row count, and flag-count table."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 4)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        prices_rows += [_make_price_row("BBB", "LSE", d, 20 + i) for i, d in enumerate(days)]
        flags_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "STALE_PRICE", "severity": "medium", "details": "d1"},
            {"ticker": "BBB", "date": days[2], "flag_type": "OHLC_INCONSISTENCY", "severity": "high", "details": "d2"},
        ]
        _setup_db(db_path, prices_rows, flags_rows)

        report = build_report(db_path)

        assert f"Date range: {days[0]} to {days[-1]}" in report
        assert "Tickers: AAA, BBB" in report
        assert "Total rows: 8" in report
        assert "| STALE_PRICE | medium | 1 |" in report
        assert "| OHLC_INCONSISTENCY | high | 1 |" in report

    def test_ticker_section_shows_explanation_when_present(self, tmp_path):
        """A flag with a matching row in `explanations` should show that explanation text."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 3)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        flags_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "OUTLIER_RETURN", "severity": "medium", "details": "big move"},
        ]
        explanations_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "OUTLIER_RETURN", "explanation": "Earnings-driven move."},
        ]
        _setup_db(db_path, prices_rows, flags_rows, explanations_rows)

        report = build_report(db_path)

        assert "## AAA" in report
        assert f"### {days[1]} — OUTLIER_RETURN (medium)" in report
        assert "- Details: big move" in report
        assert "- Explanation: Earnings-driven move." in report

    def test_ticker_section_shows_holiday_note_for_unexplained_missing_dates(self, tmp_path):
        """A MISSING_DATES flag with no explanation row should show the market-holiday fallback note."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 3)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        flags_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "MISSING_DATES", "severity": "low", "details": "gap"},
        ]
        _setup_db(db_path, prices_rows, flags_rows)

        report = build_report(db_path)

        assert "- Explanation: Excluded: known market holiday" in report

    def test_ticker_section_shows_generic_note_for_other_unexplained_flags(self, tmp_path):
        """A non-MISSING_DATES flag with no explanation row should show the generic fallback note."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 3)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        flags_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "DUPLICATE_ROW", "severity": "high", "details": "dup"},
        ]
        _setup_db(db_path, prices_rows, flags_rows)

        report = build_report(db_path)

        assert "- Explanation: No explanation available" in report

    def test_ticker_with_no_flags_shows_placeholder(self, tmp_path):
        """A ticker present in prices but absent from flags should show a 'no flags' note, not be skipped."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 3)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        _setup_db(db_path, prices_rows, flags_rows=[])

        report = build_report(db_path)

        assert "## AAA" in report
        assert "No flags for this ticker." in report

    def test_pass_rate_reflects_flagged_vs_total_rows(self, tmp_path):
        """The closing pass-rate percentage should equal (rows with no flag) / (total rows) * 100."""
        db_path = tmp_path / "market.duckdb"
        days = _weekdays(date(2026, 1, 5), 4)
        prices_rows = [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)]
        flags_rows = [
            {"ticker": "AAA", "date": days[1], "flag_type": "STALE_PRICE", "severity": "medium", "details": "d1"},
        ]
        _setup_db(db_path, prices_rows, flags_rows)

        report = build_report(db_path)

        # 1 of 4 rows flagged -> 75% passed
        assert "75.0% of rows passed all validation checks." in report

    def test_save_report_writes_file(self, tmp_path):
        """save_report() should write the given string to disk, creating parent dirs as needed."""
        output_path = tmp_path / "nested" / "report.md"

        save_report("# Test Report\n\nHello.", output_path)

        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "# Test Report\n\nHello."

    def test_default_report_filename_matches_for_identical_scope(self, tmp_path):
        """Two databases covering the same tickers and date range should hash to the same filename."""
        days = _weekdays(date(2026, 1, 5), 3)

        db_path_a = tmp_path / "a.duckdb"
        _setup_db(db_path_a, [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)], [])

        db_path_b = tmp_path / "b.duckdb"
        _setup_db(db_path_b, [_make_price_row("AAA", "NYSE", d, 99 + i) for i, d in enumerate(days)], [])

        report_date = date(2026, 7, 27)
        assert default_report_filename(db_path_a, report_date) == default_report_filename(db_path_b, report_date)

    def test_default_report_filename_differs_for_different_tickers(self, tmp_path):
        """A different ticker set covering the same date range should hash to a different filename."""
        days = _weekdays(date(2026, 1, 5), 3)

        db_path_a = tmp_path / "a.duckdb"
        _setup_db(db_path_a, [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)], [])

        db_path_b = tmp_path / "b.duckdb"
        _setup_db(db_path_b, [_make_price_row("BBB", "NYSE", d, 10 + i) for i, d in enumerate(days)], [])

        report_date = date(2026, 7, 27)
        assert default_report_filename(db_path_a, report_date) != default_report_filename(db_path_b, report_date)

    def test_default_report_filename_differs_for_different_date_range(self, tmp_path):
        """The same ticker over a different date range should hash to a different filename."""
        days_a = _weekdays(date(2026, 1, 5), 3)
        days_b = _weekdays(date(2026, 2, 2), 3)

        db_path_a = tmp_path / "a.duckdb"
        _setup_db(db_path_a, [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days_a)], [])

        db_path_b = tmp_path / "b.duckdb"
        _setup_db(db_path_b, [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days_b)], [])

        report_date = date(2026, 7, 27)
        assert default_report_filename(db_path_a, report_date) != default_report_filename(db_path_b, report_date)

    def test_default_report_filename_includes_report_date(self, tmp_path):
        """The filename should embed the given report_date regardless of the data's own date range."""
        db_path = tmp_path / "a.duckdb"
        days = _weekdays(date(2026, 1, 5), 3)
        _setup_db(db_path, [_make_price_row("AAA", "NYSE", d, 10 + i) for i, d in enumerate(days)], [])

        filename = default_report_filename(db_path, date(2026, 12, 25))

        assert filename.startswith("market_data_quality_report_2026-12-25_")
        assert filename.endswith(".md")
