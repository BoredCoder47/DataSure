
import pandas as pd

from modules.quality_checks import (
    detect_missing_values,
    detect_duplicate_rows,
    detect_iqr_outliers,
)

from modules.data_validation import (
    detect_inconsistent_categories,
    validate_numeric_ranges,
    detect_invalid_dates,
)

def test_missing_value_detection():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya"],
        "Age": [22, None],
    })

    result = detect_missing_values(df)

    age_result = result[result["column"] == "Age"].iloc[0]

    assert age_result["missing_count"] == 1
    assert age_result["missing_percentage"] == 50.0


def test_duplicate_detection():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya", "Aarav"],
        "Age": [22, 24, 22],
    })

    duplicates = detect_duplicate_rows(df)

    assert len(duplicates) == 1


def test_iqr_outlier_detection():
    df = pd.DataFrame({
        "Salary": [
            25000, 28000, 30000, 32000,
            27000, 29000, 31000, 950000,
        ]
    })

    results = detect_iqr_outliers(df)

    assert any(
        item["row_index"] == 7
        for item in results["Salary"]
    )


def test_inconsistent_categories():
    df = pd.DataFrame({
        "City": ["Mumbai", "mumbai", "Pune", " Mumbai "]
    })

    result = detect_inconsistent_categories(df)

    assert len(result) == 1
    assert result.iloc[0]["column"] == "City"


def test_numeric_range_validation():
    df = pd.DataFrame({
        "Age": [-5, 22, 130, None]
    })

    result = validate_numeric_ranges(
        df,
        {"Age": {"min": 0, "max": 120}},
    )

    assert len(result) == 2
    assert set(result["row_index"]) == {0, 2}


def test_invalid_dates():
    df = pd.DataFrame({
        "JoinDate": [
            "2025-01-10",
            "not-a-date",
            None,
            "2025-03-15",
        ]
    })

    result = detect_invalid_dates(df)

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 1
