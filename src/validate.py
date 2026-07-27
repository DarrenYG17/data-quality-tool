"""Data quality validation checks for the `prices` DuckDB table.

Runs a battery of SQL-driven checks against daily OHLCV data pulled from
yfinance for airline/travel tickers, and writes any flagged rows into a
`flags` table (ticker, date, flag_type, severity, details) for downstream
review. This module only detects and describes issues in plain terms —
turning them into narrative explanations is handled separately by
explain.py.
"""

from pathlib import Path
from typing import Union

import duckdb
import pandas as pd
import pandas_market_calendars as mcal

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market_data.duckdb"

_FLAGS_COLUMNS = ["ticker", "date", "flag_type", "severity", "details"]

_FLAGS_SCHEMA = """
CREATE TABLE flags (
    ticker TEXT NOT NULL,
    date DATE,
    flag_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    details TEXT NOT NULL
)
"""


def _empty_flags() -> pd.DataFrame:
    return pd.DataFrame(columns=_FLAGS_COLUMNS)


def check_missing_dates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Flag expected trading days with no price row at all.

    For each ticker, looks up its own exchange's trading calendar (via
    pandas_market_calendars) and computes the actual trading days between
    that ticker's min and max date in `prices`, then flags any of those
    days with no matching row. A date that wasn't a trading day for the
    ticker's exchange (weekend or market holiday) is never flagged.

    Why it matters: a missing trading day is an incomplete/broken data
    pull, not a benign calendar artifact — unlike a naive "every weekday"
    check, this won't misflag real market holidays as gaps. Severity
    defaults to LOW since a single missed day is rarely a crisis, but it's
    still a hole that will silently distort anything computed over "all
    trading days" (rolling averages, return series, etc.) unless it's
    accounted for.
    """
    ticker_bounds = con.execute(
        """
        SELECT ticker, ANY_VALUE(exchange) AS exchange, MIN(date) AS min_date, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
        """
    ).fetchdf()

    if ticker_bounds.empty:
        return _empty_flags()

    existing_dates = con.execute("SELECT ticker, date FROM prices").fetchdf()
    existing_by_ticker = {
        ticker: set(pd.to_datetime(group["date"]).dt.date)
        for ticker, group in existing_dates.groupby("ticker")
    }

    rows = []
    for row in ticker_bounds.itertuples():
        calendar = mcal.get_calendar(row.exchange)
        trading_days = calendar.valid_days(start_date=row.min_date, end_date=row.max_date)
        expected_dates = {ts.date() for ts in trading_days}

        actual_dates = existing_by_ticker.get(row.ticker, set())
        for missing_date in sorted(expected_dates - actual_dates):
            rows.append({"ticker": row.ticker, "date": missing_date})

    if not rows:
        return _empty_flags()

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    # Normalize to pd.Timestamp so this check's `date` column matches the dtype
    # the other (SQL-driven) checks produce via fetchdf() — otherwise pd.concat
    # in run_validations() ends up with a mixed-type object column.
    df["date"] = pd.to_datetime(df["date"])
    df["flag_type"] = "MISSING_DATES"
    df["severity"] = "low"
    df["details"] = df["date"].apply(
        lambda d: f"No price row found for expected trading day {d.date()}."
    )
    return df[_FLAGS_COLUMNS]


def check_stale_price(con: duckdb.DuckDBPyConnection, n: int = 3) -> pd.DataFrame:
    """Flag dates where `close` has been unchanged for N+1 consecutive trading days.

    Uses a gaps-and-islands pattern: a running counter increments every time
    `close` changes from the previous day, forming groups ("streaks") of
    consecutive equal closes, then flags rows whose streak length exceeds N.

    Why it matters: a close price that stops moving for several sessions in
    a row is unusual for a liquid, actively-traded name and often signals a
    stale or repeated data pull (e.g. yfinance re-serving the last known
    price) rather than genuine market behavior. Severity defaults to MEDIUM
    — worth investigating, but low-volume tickers can legitimately go flat
    for a few days around holidays or thin trading.
    """
    n = int(n)
    query = f"""
        WITH ordered AS (
            SELECT ticker, date, close,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
            FROM prices
        ),
        grouped AS (
            SELECT *,
                   SUM(CASE WHEN prev_close IS NULL OR close != prev_close THEN 1 ELSE 0 END)
                       OVER (PARTITION BY ticker ORDER BY date
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS streak_id
            FROM ordered
        ),
        streaks AS (
            SELECT *,
                   COUNT(*) OVER (PARTITION BY ticker, streak_id ORDER BY date
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS streak_length
            FROM grouped
        )
        SELECT ticker, date, close, streak_length
        FROM streaks
        WHERE streak_length > {n}
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        return _empty_flags()

    df["flag_type"] = "STALE_PRICE"
    df["severity"] = "medium"
    df["details"] = df.apply(
        lambda r: (
            f"close unchanged at {r['close']:.2f} for {int(r['streak_length'])} "
            "consecutive trading days"
        ),
        axis=1,
    )
    return df[_FLAGS_COLUMNS]


def check_outlier_return(con: duckdb.DuckDBPyConnection, k: float = 3.0) -> pd.DataFrame:
    """Flag daily returns that deviate from a ticker's trailing 30-day mean by more than k std devs.

    Computes the daily percentage return on `adj_close`, then a trailing
    30-day rolling mean/stddev (excluding the current day) via SQL window
    functions, and flags days where the return falls more than k standard
    deviations from that rolling mean.

    Why it matters: airline/travel stocks are already volatile, but a return
    that's a genuine statistical outlier relative to the ticker's own recent
    behavior is worth a second look — it can be a bad print in the feed, an
    unadjusted split/dividend, or real news (fine, but still worth
    surfacing). Severity defaults to MEDIUM since large moves are often
    legitimate for this sector; tighten k if false positives are too
    frequent.
    """
    k = float(k)
    query = f"""
        WITH returns AS (
            SELECT ticker, date,
                   (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1) * 100
                       AS pct_return
            FROM prices
        ),
        rolling AS (
            SELECT ticker, date, pct_return,
                   AVG(pct_return) OVER (
                       PARTITION BY ticker ORDER BY date
                       ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                   ) AS rolling_mean,
                   STDDEV_SAMP(pct_return) OVER (
                       PARTITION BY ticker ORDER BY date
                       ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                   ) AS rolling_std
            FROM returns
        )
        SELECT ticker, date, pct_return, rolling_mean, rolling_std
        FROM rolling
        WHERE rolling_std IS NOT NULL
          AND rolling_std > 0
          AND ABS(pct_return - rolling_mean) > {k} * rolling_std
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        return _empty_flags()

    df["flag_type"] = "OUTLIER_RETURN"
    df["severity"] = "medium"
    df["z_score"] = (df["pct_return"] - df["rolling_mean"]) / df["rolling_std"]
    df["details"] = df.apply(
        lambda r: (
            f"daily return {r['pct_return']:.2f}% vs trailing 30-day mean "
            f"{r['rolling_mean']:.2f}% (std {r['rolling_std']:.2f}%) — "
            f"{r['z_score']:.1f} std devs from mean"
        ),
        axis=1,
    )
    return df[_FLAGS_COLUMNS]


def check_duplicate_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Flag (ticker, date) combinations that appear more than once in `prices`.

    Note: with the current ingest.py, `prices` has a PRIMARY KEY (ticker,
    date) constraint, so a true duplicate can never actually be inserted —
    this check is effectively unreachable against data written by
    write_to_duckdb(). It's kept as a safeguard in case `prices` is ever
    populated by another path that doesn't enforce that constraint.

    Why it matters: this table should have exactly one row per ticker per
    trading day. Duplicates usually mean the ingestion job re-ran without
    deduping (or a join fanned out upstream) and will silently double-count
    volume and skew any aggregation. Severity defaults to HIGH — this is a
    structural data integrity problem, not a market observation.
    """
    query = """
        SELECT ticker, date, COUNT(*) AS occurrences
        FROM prices
        GROUP BY ticker, date
        HAVING COUNT(*) > 1
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        return _empty_flags()

    df["flag_type"] = "DUPLICATE_ROW"
    df["severity"] = "high"
    df["details"] = df["occurrences"].apply(
        lambda c: f"{int(c)} duplicate rows found for this ticker/date combination"
    )
    return df[_FLAGS_COLUMNS]


def check_ohlc_inconsistency(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Flag rows where the OHLC values are internally impossible.

    The high must be >= every other price in the session and the low must
    be <= every other price; this checks for high < low/open/close or
    low > open/close.

    Why it matters: a violation means the row is corrupted at the source
    (bad feed data, or a bad join/merge during ingestion) rather than
    reflecting real market behavior. Severity defaults to HIGH — any
    calculation built on these rows (returns, ranges, volatility) will be
    wrong.
    """
    query = """
        SELECT ticker, date, open, high, low, close
        FROM prices
        WHERE high < low OR high < open OR high < close OR low > open OR low > close
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    if df.empty:
        return _empty_flags()

    def _describe(row: pd.Series) -> str:
        violations = []
        if row["high"] < row["low"]:
            violations.append(f"high ({row['high']:.2f}) < low ({row['low']:.2f})")
        if row["high"] < row["open"]:
            violations.append(f"high ({row['high']:.2f}) < open ({row['open']:.2f})")
        if row["high"] < row["close"]:
            violations.append(f"high ({row['high']:.2f}) < close ({row['close']:.2f})")
        if row["low"] > row["open"]:
            violations.append(f"low ({row['low']:.2f}) > open ({row['open']:.2f})")
        if row["low"] > row["close"]:
            violations.append(f"low ({row['low']:.2f}) > close ({row['close']:.2f})")
        return "; ".join(violations)

    df["flag_type"] = "OHLC_INCONSISTENCY"
    df["severity"] = "high"
    df["details"] = df.apply(_describe, axis=1)
    return df[_FLAGS_COLUMNS]


def run_validations(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    stale_price_n: int = 3,
    outlier_k: float = 3.0,
) -> pd.DataFrame:
    """Run all five checks against `prices` and (re)write the `flags` table.

    Drops and recreates `flags` each run so re-running validation doesn't
    accumulate duplicate flags from previous runs.
    """
    con = duckdb.connect(str(db_path))
    try:
        checks = [
            check_missing_dates(con),
            check_stale_price(con, n=stale_price_n),
            check_outlier_return(con, k=outlier_k),
            check_duplicate_rows(con),
            check_ohlc_inconsistency(con),
        ]
        all_flags = pd.concat(checks, ignore_index=True) if checks else _empty_flags()

        con.execute("DROP TABLE IF EXISTS flags")
        con.execute(_FLAGS_SCHEMA)
        con.register("new_flags", all_flags)
        con.execute("INSERT INTO flags SELECT * FROM new_flags")
    finally:
        con.close()

    return all_flags


if __name__ == "__main__":
    flags_df = run_validations()

    if flags_df.empty:
        print("No data quality issues detected.")
    else:
        summary = (
            flags_df.groupby(["flag_type", "severity"])
            .size()
            .reset_index(name="count")
            .sort_values(["flag_type", "severity"])
        )
        print(f"Total flags: {len(flags_df)}\n")
        print(summary.to_string(index=False))
