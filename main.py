"""CLI entry point: run the ticker ingestion pipeline and verify the result."""

import argparse
import sys

import duckdb

from src.ingest import DEFAULT_CONFIG_PATH, load_config, run


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV data into DuckDB.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ticker config JSON file (default: config/tickers.json)",
    )
    args = parser.parse_args()

    try:
        run(args.config)
        db_path = load_config(args.config).get("db_path", "data/market_data.duckdb")
        verify(db_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
