
import re
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


# Column names that usually indicate numeric data.
NUMERIC_NAME_HINTS = {
    "age", "ages", "salary", "income", "price", "amount",
    "quantity", "count", "height", "weight", "marks",
    "score", "percentage", "percent", "rating", "revenue",
    "profit", "loss", "balance", "distance", "duration",
    "temperature", "experience", "employees", "population",
    "year", "month", "day",
}


def detect_inconsistent_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find category values that differ only by capitalization
    or leading/trailing whitespace.
    """
    issues = []

    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
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
    Validate values against configurable numeric bounds.

    Example:
    {"Age": {"min": 0, "max": 120}}

    Missing and non-numeric values are skipped.
    """
    issues = []

    if not isinstance(rules, dict):
        raise TypeError("Rules must be provided as a dictionary.")

    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        if not isinstance(bounds, dict):
            raise TypeError(
                f"Rules for column '{column}' must be a dictionary."
            )

        minimum = bounds.get("min")
        maximum = bounds.get("max")

        if minimum is not None and maximum is not None:
            if minimum > maximum:
                raise ValueError(
                    f"Minimum cannot exceed maximum for '{column}'."
                )

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        for index, value in numeric_values.items():
            if pd.isna(value):
                continue

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
    """
    if columns is None:
        keywords = ("date", "time", "dob")
        columns = [
            column
            for column in df.columns
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

        non_empty = (
            values.notna()
            & values.astype(str).str.strip().ne("")
        )

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

def detect_data_type_mismatches(
    df: pd.DataFrame,
    expected_types: dict | None = None,
) -> pd.DataFrame:
    """
    Detect values that do not match expected column types.

    expected_types example:
    {
        "Age": "Number",
        "JoinDate": "Date",
        "Active": "Boolean",
        "Name": "Text",
    }

    Supported types:
        Auto, Text, Number, Date, Boolean

    Auto uses inferred types and column-name heuristics.
    Missing and blank values are skipped because missing values
    are handled by the missing-value checker.
    """
    import re

    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
    )

    numeric_name_hints = {
        "age", "ages", "salary", "income", "price", "amount",
        "quantity", "count", "height", "weight", "marks",
        "score", "percentage", "percent", "rating", "revenue",
        "profit", "loss", "balance", "distance", "duration",
        "temperature", "experience", "employees", "population",
        "year", "month", "day",
    }

    supported_types = {
        "auto", "text", "number", "date", "boolean"
    }

    expected_types = expected_types or {}
    issues = []

    for column in df.columns:
        series = df[column]
        dtype = series.dtype
        requested_type = str(
            expected_types.get(column, "Auto")
        ).strip().casefold()

        if requested_type not in supported_types:
            raise ValueError(
                f"Unsupported expected type '{requested_type}' "
                f"for column '{column}'."
            )

        # Ignore missing and blank values.
        non_empty_mask = series.notna()
        non_empty_mask &= series.astype(str).str.strip().ne("")
        values = series[non_empty_mask]

        if values.empty or requested_type == "text":
            continue

        # ------------------------------------------
        # DETERMINE EXPECTED TYPE
        # ------------------------------------------

        if requested_type != "auto":
            target_type = requested_type

        elif is_bool_dtype(dtype):
            target_type = "boolean"

        elif is_datetime64_any_dtype(dtype):
            target_type = "date"

        elif is_numeric_dtype(dtype):
            target_type = "number"

        else:
            normalized_name = re.sub(
                r"[^a-z0-9]",
                "",
                str(column).casefold(),
            )
            name_parts = set(
                re.findall(r"[a-z0-9]+", str(column).casefold())
            )

            if (
                normalized_name in numeric_name_hints
                or name_parts & numeric_name_hints
            ):
                target_type = "number"
            else:
                converted = pd.to_numeric(
                    values.astype(str).str.strip(),
                    errors="coerce",
                )

                has_numeric = converted.notna().any()
                has_non_numeric = converted.isna().any()

                if has_numeric and has_non_numeric:
                    target_type = "number"
                else:
                    continue

        # ------------------------------------------
        # VALIDATE EACH VALUE
        # ------------------------------------------

        for index, value in values.items():
            text_value = str(value).strip()
            valid = True
            reason = ""

            if target_type == "number":
                converted = pd.to_numeric(
                    pd.Series([text_value]),
                    errors="coerce",
                ).iloc[0]

                valid = pd.notna(converted)

                if not valid:
                    reason = "Value is not numeric"

            elif target_type == "date":
                converted = pd.to_datetime(
                    pd.Series([text_value]),
                    errors="coerce",
                ).iloc[0]

                valid = pd.notna(converted)

                if not valid:
                    reason = "Value is not a valid date"

            elif target_type == "boolean":
                normalized = text_value.casefold()

                valid_boolean_values = {
                    "true", "false", "1", "0",
                    "yes", "no", "y", "n",
                }

                valid = normalized in valid_boolean_values

                if not valid:
                    reason = "Value is not a recognized boolean"

            if not valid:
                issues.append({
                    "row_index": index,
                    "column": str(column),
                    "value": text_value,
                    "expected_type": target_type.title(),
                    "reason": reason,
                })

    return pd.DataFrame(
        issues,
        columns=[
            "row_index",
            "column",
            "value",
            "expected_type",
            "reason",
        ],
    )