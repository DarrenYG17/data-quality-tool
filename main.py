"""CLI entry point: run the ticker ingestion pipeline and verify the result."""

import argparse
import sys

import duckdb

from src.explain import run_explanations
from src.ingest import DEFAULT_CONFIG_PATH, TickerDataUnavailable, load_config, run
from src.validate import run_validations


def verify(db_path: str) -> None:
    """Print a per-ticker summary of what landed in the prices table."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT ticker, COUNT(*) AS rows, MIN(date) AS start_date, MAX(date) AS end_date
            FROM prices
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        print("No rows found in 'prices' table.")
        return

    print(f"\n{'ticker':<10}{'rows':<8}{'start_date':<14}{'end_date':<14}")
    for ticker, row_count, start_date, end_date in rows:
        print(f"{ticker:<10}{row_count:<8}{str(start_date):<14}{str(end_date):<14}")


def print_validation_summary(flags_df) -> None:
    """Print a flag count summary grouped by flag_type and severity."""
    if flags_df.empty:
        print("\nNo data quality issues detected.")
        return

    summary = (
        flags_df.groupby(["flag_type", "severity"])
        .size()
        .reset_index(name="count")
        .sort_values(["flag_type", "severity"])
    )
    print(f"\nTotal flags: {len(flags_df)}\n")
    print(summary.to_string(index=False))


def print_explanation_summary(summary: dict) -> None:
    """Print counts of newly explained vs. already-cached vs. failed flags."""
    print(
        f"\nNewly explained: {summary['newly_explained']}\n"
        f"Already cached (skipped): {summary['already_cached']}\n"
        f"Failed: {summary['failed']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV data into DuckDB.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ticker config JSON file (default: config/tickers.json)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    db_path = config.get("db_path", "data/market_data.duckdb")

    try:
        run(args.config)
    except TickerDataUnavailable as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    try:
        verify(db_path)
    except duckdb.Error as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1

    try:
        flags_df = run_validations(db_path)
    except duckdb.Error as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print_validation_summary(flags_df)

    try:
        explanation_summary = run_explanations(db_path)
    except Exception as exc:
        print(f"Explanation failed: {exc}", file=sys.stderr)
        return 1

    print_explanation_summary(explanation_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
