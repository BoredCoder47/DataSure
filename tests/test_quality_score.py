
import pandas as pd
import pytest

from modules.quality_score import (
    calculate_quality_score,
    get_quality_rating,
)


def make_clean_results():
    return (
        pd.DataFrame(columns=["column", "variants", "issue"]),
        pd.DataFrame(columns=["row_index", "column", "value", "reason"]),
        pd.DataFrame(columns=["row_index", "column", "value", "reason"]),
    )


def test_perfect_dataset_scores_100():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya", "Rahul"],
        "Age": [22, 24, 23],
    })

    categories, ranges, dates = make_clean_results()

    result = calculate_quality_score(
        df, categories, ranges, dates
    )

    assert result["overall_score"] == 100.0


def test_missing_cells_reduce_completeness():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya"],
        "Age": [22, None],
    })

    categories, ranges, dates = make_clean_results()

    result = calculate_quality_score(
        df, categories, ranges, dates
    )

    assert result["dimensions"]["completeness"] == 75.0
    assert result["issues"]["missing_cells"] == 1


def test_duplicate_rows_reduce_uniqueness():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya", "Aarav"],
        "Age": [22, 24, 22],
    })

    categories, ranges, dates = make_clean_results()

    result = calculate_quality_score(
        df, categories, ranges, dates
    )

    assert round(result["dimensions"]["uniqueness"], 2) == 66.67
    assert result["issues"]["duplicate_rows"] == 1


def test_invalid_range_reduces_validity():
    df = pd.DataFrame({
        "Age": [22, -5, 24],
    })

    categories, _, dates = make_clean_results()

    ranges = pd.DataFrame([{
        "row_index": 1,
        "column": "Age",
        "value": -5,
        "reason": "Below minimum (0)",
    }])

    result = calculate_quality_score(
        df, categories, ranges, dates
    )

    assert round(result["dimensions"]["validity"], 2) == 66.67


def test_empty_dataset_is_rejected():
    df = pd.DataFrame(columns=["Name", "Age"])

    categories, ranges, dates = make_clean_results()

    with pytest.raises(ValueError):
        calculate_quality_score(
            df, categories, ranges, dates
        )


def test_quality_ratings():
    assert get_quality_rating(95) == "Excellent"
    assert get_quality_rating(80) == "Good"
    assert get_quality_rating(65) == "Fair"
    assert get_quality_rating(40) == "Poor"
