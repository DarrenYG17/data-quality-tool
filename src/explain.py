"""Generate LLM explanations for unexplained data-quality flags.

Connects to the DuckDB `flags` table produced by validate.py, finds flags
that don't yet have a matching row in `explanations` (via a SQL join, not a
per-row Python existence check), and asks Claude to hypothesize what likely
caused each one — using a window of surrounding price/volume history as
context. Detection and plain-terms description of issues happens in
validate.py; this module only narrates them.
"""

import json
import logging
from pathlib import Path
from typing import Union

import anthropic
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market_data.duckdb"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "explain.json"

_EXPLANATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS explanations (
    ticker TEXT NOT NULL,
    date DATE,
    flag_type TEXT NOT NULL,
    explanation TEXT NOT NULL
)
"""

_CANDIDATE_HYPOTHESES = """\
- A genuine market move (news, earnings, sector-wide event affecting airline/travel stocks)
- An unadjusted stock split or dividend
- A data provider glitch (bad print, wrong field, duplicate pull from yfinance)
- A stale or duplicate data artifact (e.g. repeated last-known price)\
"""


def load_model_name(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> str:
    """Read the Claude model name to use for explanations from a JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = config.get("model")
    if not model:
        raise ValueError(f"Config file {path} must define a 'model' string.")

    return model


def get_unexplained_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Find flags with no matching row in `explanations`, via a SQL join."""
    query = """
        SELECT f.ticker, f.date, f.flag_type, f.severity, f.details
        FROM flags f
        LEFT JOIN explanations e
          ON f.ticker = e.ticker AND f.date = e.date AND f.flag_type = e.flag_type
        WHERE e.ticker IS NULL
        ORDER BY f.ticker, f.date
    """
    return con.execute(query).fetchdf()


def get_price_window(
    con: duckdb.DuckDBPyConnection, ticker: str, date, window: int = 10
) -> pd.DataFrame:
    """Fetch `window` trading days of price/volume history on either side of `date` for `ticker`."""
    query = """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY date) AS rn
            FROM prices
            WHERE ticker = ?
        ),
        target AS (
            SELECT rn FROM ranked WHERE date = ?
        )
        SELECT ranked.date, ranked.open, ranked.high, ranked.low, ranked.close,
               ranked.volume, ranked.adj_close
        FROM ranked, target
        WHERE ranked.rn BETWEEN target.rn - ? AND target.rn + ?
        ORDER BY ranked.date
    """
    return con.execute(query, [ticker, date, window, window]).fetchdf()


def build_prompt(flag_row: pd.Series, price_window: pd.DataFrame) -> str:
    """Build a prompt giving Claude the flag details plus surrounding price/volume context."""
    history_lines = "\n".join(
        f"{row.date}  open={row.open:.2f} high={row.high:.2f} low={row.low:.2f} "
        f"close={row.close:.2f} volume={int(row.volume) if pd.notna(row.volume) else 'NA'} "
        f"adj_close={row.adj_close:.2f}"
        for row in price_window.itertuples()
    )

    return f"""A data quality check flagged the following issue in a daily OHLCV dataset:

Ticker: {flag_row['ticker']}
Date: {flag_row['date']}
Flag type: {flag_row['flag_type']}
Severity: {flag_row['severity']}
Description: {flag_row['details']}

Here is the surrounding trading-day price/volume history for this ticker (roughly 10 trading days before and after the flagged date):

{history_lines}

Briefly weigh these candidate explanations against the data above:
{_CANDIDATE_HYPOTHESES}

Then commit to the single hypothesis you find most likely and explain why.

Respond in plain prose only: no markdown, no bold text, no bullet points, no headers. Write 2-4 sentences as a single flowing paragraph — state your reasoning briefly, then your conclusion, in ordinary sentences."""


def explain_flag(
    client: anthropic.Anthropic, model: str, flag_row: pd.Series, price_window: pd.DataFrame
) -> str:
    """Call Claude to generate a narrative explanation for one flagged row."""
    prompt = build_prompt(flag_row, price_window)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.stop_reason)
    return next(block.text for block in response.content if block.type == "text")


def run_explanations(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> dict:
    """Explain all currently-unexplained flags and append results to `explanations`."""
    model = load_model_name(config_path)
    client = anthropic.Anthropic()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(_EXPLANATIONS_SCHEMA)

        total_flags = con.execute("SELECT COUNT(*) FROM flags").fetchone()[0]
        unexplained = get_unexplained_flags(con)
        already_cached = total_flags - len(unexplained)

        newly_explained = 0
        failed = 0
        rows = []

        for _, flag_row in unexplained.iterrows():
            price_window = get_price_window(con, flag_row["ticker"], flag_row["date"])
            try:
                explanation = explain_flag(client, model, flag_row, price_window)
            except Exception as exc:
                logger.error(
                    "Failed to explain %s/%s/%s: %s",
                    flag_row["ticker"], flag_row["date"], flag_row["flag_type"], exc,
                )
                failed += 1
                continue

            rows.append(
                {
                    "ticker": flag_row["ticker"],
                    "date": flag_row["date"],
                    "flag_type": flag_row["flag_type"],
                    "explanation": explanation,
                }
            )
            newly_explained += 1

        if rows:
            new_explanations = pd.DataFrame(rows)
            con.register("new_explanations", new_explanations)
            con.execute("INSERT INTO explanations SELECT * FROM new_explanations")
    finally:
        con.close()

    return {
        "newly_explained": newly_explained,
        "already_cached": already_cached,
        "failed": failed,
    }


if __name__ == "__main__":
    summary = run_explanations()
    print(
        f"Newly explained: {summary['newly_explained']}\n"
        f"Already cached (skipped): {summary['already_cached']}\n"
        f"Failed: {summary['failed']}"
    )
