
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

    Completeness:
        Percentage of non-missing cells.

    Uniqueness:
        Percentage of rows that are not duplicates.

    Validity:
        Percentage of cells without detected numeric-range
        or date violations. Each row/column location is
        counted at most once.

    Consistency:
        Percentage of distinct categorical values not
        involved in detected capitalization/whitespace issues.

    Returns:
        Overall score, dimension scores, normalized weights,
        and issue counts.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot score an empty dataset.")

    # ----------------------------------------------
    # VALIDATE WEIGHTS
    # ----------------------------------------------

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    elif not isinstance(weights, dict):
        raise TypeError("Weights must be provided as a dictionary.")

    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(
            "Weights must include completeness, uniqueness, "
            "validity, and consistency."
        )

    for name, value in weights.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Weight for '{name}' must be numeric."
            )

        if not pd.notna(value) or value < 0:
            raise ValueError(
                f"Weight for '{name}' must be non-negative "
                "and finite."
            )

    weight_total = sum(weights.values())

    if not pd.notna(weight_total) or weight_total <= 0:
        raise ValueError("At least one weight must be positive.")

    normalized_weights = {
        name: value / weight_total
        for name, value in weights.items()
    }

    # ----------------------------------------------
    # 1. COMPLETENESS
    # ----------------------------------------------

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    completeness = (
        100.0 * (total_cells - missing_cells) / total_cells
        if total_cells
        else 100.0
    )

    # ----------------------------------------------
    # 2. UNIQUENESS
    # ----------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    uniqueness = (
        100.0 * (len(df) - duplicate_rows) / len(df)
        if len(df)
        else 100.0
    )

    # ----------------------------------------------
    # 3. VALIDITY
    # ----------------------------------------------
    # Numeric-range and date issues share the same
    # row/column format. Count each affected cell once.

    invalid_locations = set()

    for issues in (range_issues, date_issues):
        if issues is None or issues.empty:
            continue

        required_columns = {"row_index", "column"}

        if not required_columns.issubset(issues.columns):
            continue

        for row, column in zip(
            issues["row_index"],
            issues["column"],
        ):
            invalid_locations.add((row, column))

    invalid_location_count = len(invalid_locations)

    validity = (
        100.0
        * max(0, total_cells - invalid_location_count)
        / total_cells
        if total_cells
        else 100.0
    )

    # ----------------------------------------------
    # 4. CONSISTENCY
    # ----------------------------------------------
    # Count distinct affected category values by column.

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
                        (str(issue["column"]), str(variant))
                    )

    total_category_values = 0

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        total_category_values += int(
            df[column].nunique(dropna=True)
        )

    consistency = (
        100.0
        * max(
            0,
            total_category_values - len(affected_categories),
        )
        / total_category_values
        if total_category_values
        else 100.0
    )

    # ----------------------------------------------
    # DIMENSION SCORES
    # ----------------------------------------------

    dimensions = {
        "completeness": _clamp_score(completeness),
        "uniqueness": _clamp_score(uniqueness),
        "validity": _clamp_score(validity),
        "consistency": _clamp_score(consistency),
    }

    # ----------------------------------------------
    # OVERALL WEIGHTED SCORE
    # ----------------------------------------------

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
            "invalid_locations": invalid_location_count,
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