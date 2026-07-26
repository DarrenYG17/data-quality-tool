"""Validation checks for assessing dataset quality."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Issue:
    check: str
    column: Optional[str]
    severity: str  # "error", "warning", "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def check_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> list[Issue]:
    issues = []
    missing_ratio = df.isna().mean()
    for column, ratio in missing_ratio.items():
        if ratio > threshold:
            issues.append(
                Issue(
                    check="missing_values",
                    column=column,
                    severity="error" if ratio > 0.5 else "warning",
                    message=f"Column '{column}' has {ratio:.1%} missing values.",
                    details={
                        "missing_ratio": round(float(ratio), 4),
                        "missing_count": int(df[column].isna().sum()),
                    },
                )
            )
    return issues


def check_duplicate_rows(df: pd.DataFrame) -> list[Issue]:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []
    return [
        Issue(
            check="duplicate_rows",
            column=None,
            severity="warning",
            message=f"Found {dup_count} duplicate row(s).",
            details={"duplicate_count": dup_count},
        )
    ]


def check_duplicate_columns(df: pd.DataFrame) -> list[Issue]:
    issues = []
    seen: dict[tuple, str] = {}
    for column in df.columns:
        key = tuple(df[column].fillna("__NA__").tolist())
        if key in seen:
            issues.append(
                Issue(
                    check="duplicate_columns",
                    column=column,
                    severity="warning",
                    message=f"Column '{column}' duplicates column '{seen[key]}'.",
                    details={"duplicate_of": seen[key]},
                )
            )
        else:
            seen[key] = column
    return issues


def check_constant_columns(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if unique_count <= 1:
            issues.append(
                Issue(
                    check="constant_column",
                    column=column,
                    severity="info",
                    message=f"Column '{column}' has a single distinct value (or is empty).",
                    details={"unique_count": int(unique_count)},
                )
            )
    return issues


def check_outliers(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> list[Issue]:
    issues = []
    numeric_df = df.select_dtypes(include="number")
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers = series[(series < lower) | (series > upper)]
        if not outliers.empty:
            issues.append(
                Issue(
                    check="outliers",
                    column=column,
                    severity="warning",
                    message=(
                        f"Column '{column}' has {len(outliers)} potential outlier(s) "
                        f"outside [{lower:.2f}, {upper:.2f}]."
                    ),
                    details={
                        "outlier_count": int(len(outliers)),
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                    },
                )
            )
    return issues


def run_validations(
    df: pd.DataFrame,
    missing_threshold: float = 0.0,
    iqr_multiplier: float = 1.5,
) -> list[Issue]:
    """Run the full suite of data quality checks and return all issues found."""
    issues: list[Issue] = []
    issues += check_missing_values(df, threshold=missing_threshold)
    issues += check_duplicate_rows(df)
    issues += check_duplicate_columns(df)
    issues += check_constant_columns(df)
    issues += check_outliers(df, iqr_multiplier=iqr_multiplier)
    return issues
