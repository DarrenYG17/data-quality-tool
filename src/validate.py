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
    """
    Build an empty, correctly-columned flags DataFrame.

    @return: An empty DataFrame with columns [ticker, date, flag_type, severity, details].
    """
    return pd.DataFrame(columns=_FLAGS_COLUMNS)


def check_missing_dates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Flag expected trading days that have no price row at all.

    @param con: Open DuckDB connection with a `prices` table containing
        (ticker, exchange, date, ...) rows.
    @return: A flags DataFrame (possibly empty) with one row per missing
        trading day, flag_type "MISSING_DATES", severity "low".
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
        # Each ticker gets its own exchange's real trading calendar (not a naive
        # "every weekday" assumption), so a market holiday is never misflagged as a gap.
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
    # LOW: most gaps turn out to be routine holidays that slipped past the
    # calendar check (or a single incomplete pull) rather than pipeline failures —
    # this is about visibility, not alarm.
    df["severity"] = "low"
    df["details"] = df["date"].apply(
        lambda d: f"No price row found for expected trading day {d.date()}."
    )
    return df[_FLAGS_COLUMNS]


def check_stale_price(con: duckdb.DuckDBPyConnection, n: int = 3) -> pd.DataFrame:
    """
    Flag dates where `close` has been unchanged for more than n consecutive trading days.

    @param con: Open DuckDB connection with a `prices` table.
    @param n: Streak-length threshold; a row is flagged once its run of
        identical consecutive closes exceeds n.
    @return: A flags DataFrame (possibly empty), flag_type "STALE_PRICE",
        severity "medium".
    """
    n = int(n)
    query = f"""
        WITH ordered AS (
            SELECT ticker, date, close,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
            FROM prices
        ),
        grouped AS (
            -- Gaps-and-islands trick: increment a running counter every time close
            -- actually changes, so all rows sharing a value form one "streak_id" group.
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
    # MEDIUM: unusual for a liquid, actively-traded name and often signals a
    # stale/repeated data pull, but low-volume tickers can legitimately go flat
    # for a few days around holidays or thin trading.
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
    """
    Flag daily returns that deviate from a ticker's own trailing 30-day mean by more than k std devs.

    @param con: Open DuckDB connection with a `prices` table.
    @param k: Number of standard deviations from the trailing rolling mean
        beyond which a day's return is considered an outlier.
    @return: A flags DataFrame (possibly empty), flag_type "OUTLIER_RETURN",
        severity "medium".
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
            -- "30 PRECEDING AND 1 PRECEDING" excludes the current day itself, so today's
            -- return is judged against the ticker's own recent history, not against itself.
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
    # MEDIUM: airline/travel stocks are already volatile, so a large move is often
    # legitimate news rather than a data error; tighten k if false positives pile up.
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
    """
    Flag (ticker, date) combinations that appear more than once in `prices`.

    Note: with the current ingest.py, `prices` has a PRIMARY KEY (ticker,
    date) constraint, so a true duplicate can never actually be inserted —
    this check is effectively unreachable against data written by
    write_to_duckdb(). It's kept as a safeguard in case `prices` is ever
    populated by another path that doesn't enforce that constraint.

    @param con: Open DuckDB connection with a `prices` table.
    @return: A flags DataFrame (possibly empty), flag_type "DUPLICATE_ROW",
        severity "high".
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
    # HIGH: this table should have exactly one row per ticker per trading day;
    # duplicates silently double-count volume and skew any aggregation.
    df["severity"] = "high"
    df["details"] = df["occurrences"].apply(
        lambda c: f"{int(c)} duplicate rows found for this ticker/date combination"
    )
    return df[_FLAGS_COLUMNS]


def check_ohlc_inconsistency(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Flag rows where the OHLC values are internally impossible.

    @param con: Open DuckDB connection with a `prices` table.
    @return: A flags DataFrame (possibly empty), flag_type "OHLC_INCONSISTENCY",
        severity "high", with `details` listing every specific violation found
        (a row can fail more than one condition at once).
    """
    # high must be >= every other price in the session and low must be <= every
    # other price; any violation of that means the row is internally impossible.
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
        """
        Build a human-readable list of every OHLC constraint this row violates.

        @param row: A row from the query above (open, high, low, close present).
        @return: A "; "-joined string of each specific violation (a row can fail
            more than one condition at once).
        """
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
    # HIGH: a violation means the row is corrupted at the source, not reflecting
    # real market behavior — any calculation built on it will be wrong.
    df["severity"] = "high"
    df["details"] = df.apply(_describe, axis=1)
    return df[_FLAGS_COLUMNS]


def run_validations(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    stale_price_n: int = 3,
    outlier_k: float = 3.0,
) -> pd.DataFrame:
    """
    Run all five checks against `prices` and (re)write the `flags` table.

    @param db_path: Path to the DuckDB database file.
    @param stale_price_n: Streak-length threshold passed through to check_stale_price().
    @param outlier_k: Std-dev threshold passed through to check_outlier_return().
    @return: The combined flags DataFrame written to the `flags` table (all
        five checks concatenated).
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

        # Drop/recreate rather than append, so re-running validation on the same
        # data doesn't accumulate duplicate flags from previous runs.
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
