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

_CANDIDATE_HYPOTHESES = (
    "a genuine market move, an unadjusted stock split or dividend, "
    "a data provider glitch, or a stale/duplicate data artifact"
)


def load_model_name(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> str:
    """
    Read the Claude model name to use for explanations from a JSON config file.

    @param config_path: Path to the JSON config file; must define a `model` string.
    @return: The model name (e.g. "claude-sonnet-5").
    @raise FileNotFoundError: if `config_path` does not exist.
    @raise ValueError: if the config has no (or an empty) `model` field.
    """
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
    """
    Find flags with no matching row in `explanations`.

    @param con: Open DuckDB connection with `flags` and `explanations` tables.
    @return: DataFrame of flags (ticker, date, flag_type, severity, details)
        that don't yet have an explanation, via a LEFT JOIN + IS NULL filter
        (not a per-row Python existence check).
    """
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
    """
    Fetch `window` trading days of price/volume history on either side of `date` for `ticker`.

    @param con: Open DuckDB connection with a `prices` table.
    @param ticker: Ticker symbol to fetch history for.
    @param date: The flagged date to center the window on. Note: if this
        exact date has no row in `prices` (as is always true for a
        MISSING_DATES flag), the `target` CTE below finds no rank to anchor
        on, and this returns an *empty* DataFrame rather than the nearest
        available days — callers should not assume a non-empty result.
    @param window: Number of trading days to include on each side of `date`.
    @return: DataFrame of (date, open, high, low, close, volume, adj_close)
        rows, ordered chronologically.
    """
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
    """
    Build the prompt giving Claude the flag details plus surrounding price/volume context.

    @param flag_row: A single row (ticker, date, flag_type, severity, details)
        from get_unexplained_flags().
    @param price_window: The corresponding price/volume history from get_price_window().
    @return: The full prompt string to send to the Claude API.
    """
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

Briefly consider what's most likely ({_CANDIDATE_HYPOTHESES}), then state your conclusion in 2-3 sentences total. Do not address every hypothesis individually — just reach and justify your best explanation, in plain prose with no markdown, bold text, bullet points, or headers."""


def explain_flag(
    client: anthropic.Anthropic, model: str, flag_row: pd.Series, price_window: pd.DataFrame
) -> str:
    """
    Call Claude to generate a narrative explanation for one flagged row.

    @param client: An initialized Anthropic client.
    @param model: Model name to use for the request (from load_model_name()).
    @param flag_row: A single row from get_unexplained_flags().
    @param price_window: The corresponding price/volume history from get_price_window().
    @return: The plain-text explanation Claude returned.
    @raise Exception: propagates any error from the Anthropic API call
        (network, auth, rate limit, etc.) — callers are expected to catch
        and log per-flag rather than let one failure abort a whole batch.
    """
    prompt = build_prompt(flag_row, price_window)
    # Thinking is disabled because this is a short, bounded classification task with
    # no tools involved — leaving it on would eat into max_tokens invisibly (thinking
    # + visible text share the same budget) and could truncate the answer.
    response = client.messages.create(
        model=model,
        max_tokens=500,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def run_explanations(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> dict:
    """
    Explain all currently-unexplained flags and append results to `explanations`.

    @param db_path: Path to the DuckDB database file.
    @param config_path: Path to the JSON config file naming the Claude model to use.
    @return: dict with keys "newly_explained", "already_cached", and "failed" counts.
    @raise FileNotFoundError: propagated from load_model_name() if the config is missing.
    @raise ValueError: propagated from load_model_name() if the config is malformed.
    """
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
                # Catch broadly and continue: one flag's API failure (rate limit, auth,
                # network) should never abort explaining the rest of the batch.
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
