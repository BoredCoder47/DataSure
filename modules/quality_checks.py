
import pandas as pd


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values in every column."""

    counts = df.isna().sum()

    result = pd.DataFrame({
        "column": counts.index,
        "missing_count": counts.values,
    })

    result["missing_percentage"] = (
        result["missing_count"] / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    return result.sort_values(
        "missing_count", ascending=False
    ).reset_index(drop=True)


def detect_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate rows, excluding their first occurrence."""

    return df.loc[df.duplicated(keep="first")].copy()


def detect_iqr_outliers(df: pd.DataFrame) -> dict:
    """Detect numerical outliers using the IQR rule."""

    results = {}

    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            results[column] = []
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (numeric_df[column] < lower_bound)
            | (numeric_df[column] > upper_bound)
        )

        results[column] = [
            {
                "row_index": int(index),
                "value": float(numeric_df.loc[index, column]),
            }
            for index in numeric_df.index[mask.fillna(False)]
        ]

    return results
