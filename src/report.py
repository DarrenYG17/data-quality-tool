"""Assemble a markdown data quality report from the DuckDB `prices`/`flags`/`explanations` tables.

Pulls together what validate.py flagged and what explain.py narrated into a
single human-readable report: an overall summary, a per-ticker breakdown of
every flag in chronological order, and a closing pass-rate figure.
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market_data.duckdb"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

_HOLIDAY_NOTE = "Excluded: known market holiday"
_NO_EXPLANATION_NOTE = "No explanation available"


def _build_summary_section(con: duckdb.DuckDBPyConnection) -> str:
    """
    Build the report's summary section: date range, tickers, row count, and flag-count table.

    @param con: Open DuckDB connection with `prices` and `flags` tables.
    @return: Markdown string for the "## Summary" section.
    """
    min_date, max_date, total_rows = con.execute(
        "SELECT MIN(date), MAX(date), COUNT(*) FROM prices"
    ).fetchone()

    tickers = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchdf()["ticker"].tolist()

    flag_counts = con.execute(
        """
        SELECT flag_type, severity, COUNT(*) AS count
        FROM flags
        GROUP BY flag_type, severity
        ORDER BY flag_type, severity
        """
    ).fetchdf()

    lines = [
        "## Summary",
        "",
        f"- Date range: {min_date} to {max_date}",
        f"- Tickers: {', '.join(tickers)}",
        f"- Total rows: {total_rows}",
        "",
    ]

    if flag_counts.empty:
        lines.append("No flags recorded.")
    else:
        lines.append("| Flag Type | Severity | Count |")
        lines.append("|---|---|---|")
        for row in flag_counts.itertuples():
            lines.append(f"| {row.flag_type} | {row.severity} | {row.count} |")

    lines.append("")
    return "\n".join(lines)


def _build_ticker_sections(con: duckdb.DuckDBPyConnection) -> str:
    """
    Build one markdown section per ticker, listing its flags in chronological order.

    @param con: Open DuckDB connection with `prices`, `flags`, and `explanations` tables.
    @return: Markdown string covering every ticker present in `prices`
        (including those with zero flags, which get a short placeholder line).
        Each flag shows its `details` plus the matching `explanations` text if
        one exists; otherwise a fallback note (the market-holiday note for
        MISSING_DATES, a generic "no explanation" note otherwise).
    """
    tickers = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchdf()["ticker"].tolist()

    lines = []
    for ticker in tickers:
        lines.append(f"## {ticker}")
        lines.append("")

        flags = con.execute(
            """
            SELECT f.date, f.flag_type, f.severity, f.details, e.explanation
            FROM flags f
            LEFT JOIN explanations e
              ON f.ticker = e.ticker AND f.date = e.date AND f.flag_type = e.flag_type
            WHERE f.ticker = ?
            ORDER BY f.date
            """,
            [ticker],
        ).fetchdf()

        if flags.empty:
            lines.append("No flags for this ticker.")
            lines.append("")
            continue

        for row in flags.itertuples():
            lines.append(f"### {row.date.date()} — {row.flag_type} ({row.severity})")
            lines.append(f"- Details: {row.details}")
            if pd.notna(row.explanation):
                lines.append(f"- Explanation: {row.explanation}")
            elif row.flag_type == "MISSING_DATES":
                # MISSING_DATES flags are the common case with no LLM explanation: the
                # calendar-driven check already screens out holidays before flagging,
                # so a note is more honest than forcing an LLM call with no price data
                # to reason over (see get_price_window()'s docstring in explain.py).
                lines.append(f"- Explanation: {_HOLIDAY_NOTE}")
            else:
                lines.append(f"- Explanation: {_NO_EXPLANATION_NOTE}")
            lines.append("")

    return "\n".join(lines)


def _build_pass_rate_section(con: duckdb.DuckDBPyConnection) -> str:
    """
    Build the closing section stating what percentage of rows had zero flags.

    @param con: Open DuckDB connection with `prices` and `flags` tables.
    @return: Markdown string for the "## Result" section, framed positively
        (percentage that passed, not percentage that failed).
    """
    total_rows = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    flagged_rows = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT ticker, date FROM flags)"
    ).fetchone()[0]

    pass_rate = 100.0 * (total_rows - flagged_rows) / total_rows if total_rows else 0.0

    return f"## Result\n\n{pass_rate:.1f}% of rows passed all validation checks.\n"


def build_report(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> str:
    """
    Assemble the full markdown report from the DuckDB database.

    @param db_path: Path to the DuckDB database file (opened read-only).
    @return: The complete report as a single markdown string (title +
        summary + per-ticker sections + pass-rate result).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        sections = [
            "# Data Quality Report",
            "",
            _build_summary_section(con),
            _build_ticker_sections(con),
            _build_pass_rate_section(con),
        ]
    finally:
        con.close()

    return "\n".join(sections)


def save_report(report: str, path: Union[str, Path]) -> None:
    """
    Write the markdown report string to disk.

    @param report: The report text to write (typically build_report()'s output).
    @param path: Destination file path; parent directories are created as needed.
    @return: None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _ticker_date_range_hash(con: duckdb.DuckDBPyConnection) -> str:
    """
    Compute a short hash identifying the ticker set and date range covered by `prices`.

    @param con: Open DuckDB connection with a `prices` table.
    @return: An 8-character hex hash. Two databases covering the same tickers
        and date range hash identically (so a rerun overwrites the same
        filename); a different scope (different tickers or date range)
        hashes differently, avoiding filename collisions between distinct
        reports generated on the same day.
    """
    tickers = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchdf()["ticker"].tolist()
    min_date, max_date = con.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
    basis = f"{','.join(tickers)}|{min_date}|{max_date}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:8]


def default_report_filename(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    report_date: Optional[date] = None,
) -> str:
    """
    Build the default report filename for a given database's current scope.

    @param db_path: Path to the DuckDB database file (opened read-only to
        compute the content hash).
    @param report_date: Date to embed in the filename; defaults to today.
    @return: Filename of the form "market_data_quality_report_<date>_<hash>.md".
    """
    report_date = report_date or date.today()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        content_hash = _ticker_date_range_hash(con)
    finally:
        con.close()

    return f"market_data_quality_report_{report_date.isoformat()}_{content_hash}.md"


if __name__ == "__main__":
    report = build_report()
    output_path = DEFAULT_REPORTS_DIR / default_report_filename()
    save_report(report, output_path)
    print(f"Report written to {output_path}")
