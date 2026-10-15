
import pandas as pd


DEFAULT_WEIGHTS = {
    "completeness": 0.30,
    "uniqueness": 0.25,
    "validity": 0.25,
    "consistency": 0.20,
}


def _clamp_score(value: float) -> float:
    """Keep a score within the range 0–100."""
    return max(0.0, min(100.0, float(value)))


def calculate_quality_score(
    df: pd.DataFrame,
    category_issues: pd.DataFrame,
    range_issues: pd.DataFrame,
    date_issues: pd.DataFrame,
    weights: dict | None = None,
) -> dict:
    """
    Calculate a heuristic dataset quality score.

    Completeness: proportion of non-missing cells.
    Uniqueness: proportion of rows that are not duplicates.
    Validity: proportion of records without detected range/date issues.
    Consistency: proportion of category values not involved in
                 detected capitalization/whitespace inconsistencies.

    Returns the overall score, dimension scores, and issue counts.
    """

    if df.empty:
        raise ValueError("Cannot score an empty dataset.")

    weights = weights or DEFAULT_WEIGHTS.copy()

    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(
            "Weights must include completeness, uniqueness, "
            "validity, and consistency."
        )

    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative.")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one weight must be positive.")

    # ----------------------------------------------
    # 1. COMPLETENESS
    # ----------------------------------------------

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    completeness = (
        100.0 * (total_cells - missing_cells) / total_cells
        if total_cells else 100.0
    )

    # ----------------------------------------------
    # 2. UNIQUENESS
    # ----------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    uniqueness = (
        100.0 * (len(df) - duplicate_rows) / len(df)
        if len(df) else 100.0
    )

    # ----------------------------------------------
    # 3. VALIDITY
    # ----------------------------------------------
    # Count distinct row/column locations with detected
    # numeric-range or date violations. A location is only
    # counted once if it appears in both results.

    invalid_locations = set()

    for issues in (range_issues, date_issues):
        if issues is None or issues.empty:
            continue

        required = {"row_index", "column"}

        if required.issubset(issues.columns):
            invalid_locations.update(
                (row, column)
                for row, column in zip(
                    issues["row_index"],
                    issues["column"],
                )
            )

    validity = (
        100.0 * max(0, total_cells - len(invalid_locations))
        / total_cells
        if total_cells else 100.0
    )

    # ----------------------------------------------
    # 4. CONSISTENCY
    # ----------------------------------------------
    # Count distinct category values involved in flagged
    # variant groups. For example, Mumbai/mumbai counts
    # as two affected values in that column.

    affected_categories = set()

    if (
        category_issues is not None
        and not category_issues.empty
        and {"column", "variants"}.issubset(category_issues.columns)
    ):
        for _, issue in category_issues.iterrows():
            variants = issue["variants"]

            if isinstance(variants, (list, tuple, set)):
                for variant in variants:
                    affected_categories.add(
                        (issue["column"], str(variant))
                    )

    total_category_values = 0

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        total_category_values += int(df[column].nunique(dropna=True))

    consistency = (
        100.0 * max(
            0, total_category_values - len(affected_categories)
        ) / total_category_values
        if total_category_values else 100.0
    )

    # ----------------------------------------------
    # OVERALL WEIGHTED SCORE
    # ----------------------------------------------

    dimensions = {
        "completeness": _clamp_score(completeness),
        "uniqueness": _clamp_score(uniqueness),
        "validity": _clamp_score(validity),
        "consistency": _clamp_score(consistency),
    }

    weight_total = sum(weights.values())

    normalized_weights = {
        name: value / weight_total
        for name, value in weights.items()
    }

    overall = sum(
        dimensions[name] * normalized_weights[name]
        for name in dimensions
    )

    return {
        "overall_score": round(_clamp_score(overall), 2),
        "dimensions": {
            name: round(score, 2)
            for name, score in dimensions.items()
        },
        "weights": normalized_weights,
        "issues": {
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "invalid_locations": len(invalid_locations),
            "inconsistent_category_values": len(
                affected_categories
            ),
        },
    }


def get_quality_rating(score: float) -> str:
    """Convert the numeric score into a readable rating."""

    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"
