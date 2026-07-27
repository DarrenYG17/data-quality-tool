"""Ingest daily OHLCV data for configured tickers via yfinance into DuckDB."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tickers.json"


class TickerDataUnavailable(RuntimeError):
    """Raised when yfinance returned no usable data for any configured ticker."""

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    exchange TEXT NOT NULL,
    date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    adj_close DOUBLE,
    PRIMARY KEY (ticker, date)
)
"""


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict:
    """Load the ticker list and settings from a JSON config file.

    Each entry in `tickers` must be an object with `symbol` and `exchange`
    fields, e.g. {"symbol": "EZJ.L", "exchange": "LSE"}.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    tickers = config.get("tickers")
    if not tickers:
        raise ValueError(f"Config file {path} must define a non-empty 'tickers' list.")

    for entry in tickers:
        if not isinstance(entry, dict) or "symbol" not in entry or "exchange" not in entry:
            raise ValueError(
                f"Config file {path}: each ticker must be an object with 'symbol' and "
                f"'exchange' fields, got {entry!r}."
            )

    return config


def fetch_ticker_history(
    ticker: str,
    period: str = "1mo",
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV history for a single ticker, retrying on rate limits or empty responses."""
    for attempt in range(1, max_retries + 1):
        try:
            history = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        except Exception as exc:
            logger.warning("Attempt %d/%d for %s failed: %s", attempt, max_retries, ticker, exc)
            history = None

        if history is not None and not history.empty:
            return history

        if attempt < max_retries:
            wait = backoff_seconds * attempt
            logger.info(
                "No data for %s on attempt %d/%d, retrying in %.1fs...",
                ticker, attempt, max_retries, wait,
            )
            time.sleep(wait)

    logger.error("Giving up on %s after %d attempt(s); no data returned.", ticker, max_retries)
    return None


def normalize_history(symbol: str, exchange: str, history: pd.DataFrame) -> pd.DataFrame:
    """Reshape a yfinance history DataFrame into the prices table's column layout."""
    available = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if c in history.columns]
    df = history.reset_index()[["Date"] + available].copy()

    rename_map = {
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
    }
    df = df.rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "volume", "adj_close"]:
        if col not in df.columns:
            df[col] = None

    df.insert(0, "ticker", symbol)
    df.insert(1, "exchange", exchange)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df[["ticker", "exchange", "date", "open", "high", "low", "close", "volume", "adj_close"]]


def fetch_all(tickers: list[dict], period: str = "1mo") -> pd.DataFrame:
    """Fetch and normalize OHLCV history for all tickers, skipping any that fail entirely.

    Each entry in `tickers` is a dict with `symbol` and `exchange` keys.
    """
    frames = []
    failed = []

    for entry in tickers:
        symbol = entry["symbol"]
        exchange = entry["exchange"]
        history = fetch_ticker_history(symbol, period=period)
        if history is None:
            failed.append(symbol)
            continue
        frames.append(normalize_history(symbol, exchange, history))

    if failed:
        logger.warning("No data retrieved for %d ticker(s): %s", len(failed), ", ".join(failed))

    if not frames:
        raise TickerDataUnavailable(
            f"No data was retrieved for any configured ticker: {', '.join(failed)}"
        )

    return pd.concat(frames, ignore_index=True)


def write_to_duckdb(df: pd.DataFrame, db_path: Union[str, Path]) -> None:
    """Create the prices table if needed and upsert rows, replacing any (ticker, date) overlap."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(_TABLE_SCHEMA)
        con.register("new_prices", df)
        con.execute(
            "DELETE FROM prices WHERE (ticker, date) IN (SELECT ticker, date FROM new_prices)"
        )
        con.execute("INSERT INTO prices SELECT * FROM new_prices")
    finally:
        con.close()


def run(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    tickers = config["tickers"]
    db_path = config.get("db_path", "data/market_data.duckdb")
    period = config.get("period", "1mo")

    df = fetch_all(tickers, period=period)
    write_to_duckdb(df, db_path)
    logger.info("Wrote %d row(s) covering %d ticker(s) to %s", len(df), df["ticker"].nunique(), db_path)


if __name__ == "__main__":
    run()
