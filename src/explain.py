"""Human-readable explanations for data quality issues."""

from dataclasses import dataclass

from src.validate import Issue

_WHY_IT_MATTERS = {
    "missing_values": (
        "Missing values can bias analysis, break downstream joins, and cause "
        "errors in models that expect complete data."
    ),
    "duplicate_rows": (
        "Duplicate rows can inflate counts, skew aggregations, and distort "
        "statistical analysis."
    ),
    "duplicate_columns": (
        "Duplicate columns waste storage and can confuse downstream consumers "
        "about which field is authoritative."
    ),
    "constant_column": (
        "A column with no variation carries no predictive or analytical value "
        "and may indicate a data collection error."
    ),
    "outliers": (
        "Outliers can indicate data entry errors, sensor faults, or genuinely "
        "rare events that deserve investigation before modeling."
    ),
}

_SUGGESTED_FIX = {
    "missing_values": (
        "Investigate the source of missingness; consider imputation, dropping "
        "the column/rows, or fixing the upstream pipeline."
    ),
    "duplicate_rows": (
        "Deduplicate using DataFrame.drop_duplicates() after confirming "
        "duplicates are unintentional."
    ),
    "duplicate_columns": (
        "Drop the redundant column, or rename it to clarify intent if the "
        "duplication is expected."
    ),
    "constant_column": (
        "Confirm whether the column is expected to be constant; otherwise "
        "check the ingestion logic."
    ),
    "outliers": (
        "Review the flagged records manually; correct erroneous values or "
        "document why they are legitimate."
    ),
}

_DEFAULT_WHY = "This may affect the reliability of downstream analysis."
_DEFAULT_FIX = "Review the flagged data manually."


@dataclass
class Explanation:
    issue: Issue
    why_it_matters: str
    suggested_fix: str


def explain_issue(issue: Issue) -> Explanation:
    return Explanation(
        issue=issue,
        why_it_matters=_WHY_IT_MATTERS.get(issue.check, _DEFAULT_WHY),
        suggested_fix=_SUGGESTED_FIX.get(issue.check, _DEFAULT_FIX),
    )


def explain_issues(issues: list[Issue]) -> list[Explanation]:
    return [explain_issue(issue) for issue in issues]
