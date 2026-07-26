"""Reporting utilities for summarizing data quality results."""

from pathlib import Path

import pandas as pd

from src.explain import Explanation

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def build_report(
    df: pd.DataFrame,
    explanations: list[Explanation],
    source_name: str = "dataset",
) -> str:
    """Render a markdown data quality report summarizing the dataset and its issues."""
    lines = [f"# Data Quality Report: {source_name}", ""]
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append(f"- Total issues found: {len(explanations)}")
    lines.append("")

    if not explanations:
        lines.append("No data quality issues detected.")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for exp in explanations:
        counts[exp.issue.severity] = counts.get(exp.issue.severity, 0) + 1
    summary = ", ".join(
        f"{count} {severity}"
        for severity, count in sorted(counts.items(), key=lambda kv: _SEVERITY_ORDER.get(kv[0], 99))
    )
    lines.append(f"**Summary:** {summary}")
    lines.append("")

    ordered = sorted(explanations, key=lambda e: _SEVERITY_ORDER.get(e.issue.severity, 99))
    for exp in ordered:
        issue = exp.issue
        column_label = f" (`{issue.column}`)" if issue.column else ""
        lines.append(f"## [{issue.severity.upper()}] {issue.check}{column_label}")
        lines.append(f"- **Issue:** {issue.message}")
        lines.append(f"- **Why it matters:** {exp.why_it_matters}")
        lines.append(f"- **Suggested fix:** {exp.suggested_fix}")
        lines.append("")

    return "\n".join(lines)


def save_report(report_text: str, output_path: str) -> None:
    """Write a report string to disk, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
