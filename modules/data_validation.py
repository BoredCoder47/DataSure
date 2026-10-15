
import pandas as pd


def detect_inconsistent_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find category values that differ only by capitalization
    or leading/trailing whitespace.

    Example: Mumbai, mumbai, " Mumbai "
    These are flagged for review, not automatically changed.
    """
    issues = []

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna()

        groups = {}

        for value in values.unique():
            normalized = str(value).strip().casefold()
            groups.setdefault(normalized, set()).add(str(value))

        for variants in groups.values():
            if len(variants) > 1:
                issues.append({
                    "column": str(column),
                    "variants": sorted(variants),
                    "issue": "Capitalization or whitespace inconsistency",
                })

    return pd.DataFrame(
        issues,
        columns=["column", "variants", "issue"],
    )


def validate_numeric_ranges(
    df: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Check numeric columns against user-defined bounds.

    Example:
    {"Age": {"min": 0, "max": 120}}
    """
    issues = []

    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[column], errors="coerce"
        )

        # Ignore missing and non-numeric values here.
        # Those should be handled by other quality checks.
        for index, value in numeric_values.items():
            if pd.isna(value):
                continue

            minimum = bounds.get("min")
            maximum = bounds.get("max")

            if minimum is not None and value < minimum:
                reason = f"Below minimum ({minimum})"
            elif maximum is not None and value > maximum:
                reason = f"Above maximum ({maximum})"
            else:
                continue

            issues.append({
                "row_index": index,
                "column": str(column),
                "value": value,
                "reason": reason,
            })

    return pd.DataFrame(
        issues,
        columns=["row_index", "column", "value", "reason"],
    )


def detect_invalid_dates(
    df: pd.DataFrame,
    columns=None,
) -> pd.DataFrame:
    """
    Detect unparseable, non-empty values in date-like columns.

    By default, columns whose names contain date, time, or
    dob are checked. Pass columns explicitly to override this.
    """
    if columns is None:
        keywords = ("date", "time", "dob")
        columns = [
            column for column in df.columns
            if any(
                word in str(column).casefold()
                for word in keywords
            )
        ]

    issues = []

    for column in columns:
        if column not in df.columns:
            continue

        values = df[column]
        non_empty = values.notna() & values.astype(str).str.strip().ne("")

        parsed = pd.to_datetime(
            values.where(non_empty),
            errors="coerce",
        )

        invalid_mask = non_empty & parsed.isna()

        for index in df.index[invalid_mask]:
            issues.append({
                "row_index": index,
                "column": str(column),
                "value": str(df.at[index, column]),
                "reason": "Unrecognized date value",
            })

    return pd.DataFrame(
        issues,
        columns=["row_index", "column", "value", "reason"],
    )
