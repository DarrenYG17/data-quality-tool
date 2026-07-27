"""Assemble a markdown data quality report from the DuckDB `prices`/`flags`/`explanations` tables.

Pulls together what validate.py flagged and what explain.py narrated into a
single human-readable report: an overall summary, a per-ticker breakdown of
every flag in chronological order, and a closing pass-rate figure.
"""

from datetime import date
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market_data.duckdb"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

_HOLIDAY_NOTE = "Excluded: known market holiday"
_NO_EXPLANATION_NOTE = "No explanation available"


def _build_summary_section(con: duckdb.DuckDBPyConnection) -> str:
    """Date range, tickers covered, total row count, and a flag-count table."""
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
    """One section per ticker, listing its flags in chronological order with explanations."""
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
                lines.append(f"- Explanation: {_HOLIDAY_NOTE}")
            else:
                lines.append(f"- Explanation: {_NO_EXPLANATION_NOTE}")
            lines.append("")

    return "\n".join(lines)


def _build_pass_rate_section(con: duckdb.DuckDBPyConnection) -> str:
    """What percentage of (ticker, date) rows in `prices` had zero flags at all."""
    total_rows = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    flagged_rows = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT ticker, date FROM flags)"
    ).fetchone()[0]

    pass_rate = 100.0 * (total_rows - flagged_rows) / total_rows if total_rows else 0.0

    return f"## Result\n\n{pass_rate:.1f}% of rows passed all validation checks.\n"


def build_report(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> str:
    """Assemble the full markdown report from the DuckDB database."""
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
    """Write the markdown report string to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    report = build_report()
    output_path = DEFAULT_REPORTS_DIR / f"report_{date.today().isoformat()}.md"
    save_report(report, output_path)
    print(f"Report written to {output_path}")
