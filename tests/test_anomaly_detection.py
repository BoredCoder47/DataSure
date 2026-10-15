
import pandas as pd
import pytest

from modules.anomaly_detection import detect_anomalies


def test_anomaly_detection_returns_expected_columns():
    df = pd.DataFrame({
        "Age": [22, 24, 23, 21, 25, 26, 27, 28],
        "Salary": [
            25000, 30000, 28000, 27000,
            32000, 29000, 31000, 950000,
        ],
    })

    result = detect_anomalies(df)

    assert set(result.columns) == {
        "row_index",
        "is_anomaly",
        "anomaly_score",
    }

    assert len(result) == len(df)
    assert result["is_anomaly"].dtype == bool


def test_anomaly_scores_are_sorted_descending():
    df = pd.DataFrame({
        "Value": [10, 11, 9, 12, 10, 11, 500],
    })

    result = detect_anomalies(df)

    assert result["anomaly_score"].is_monotonic_decreasing


def test_anomaly_detection_handles_missing_values():
    df = pd.DataFrame({
        "Age": [20, 21, None, 23, 22],
        "Salary": [20000, 22000, 21000, 23000, 21500],
    })

    result = detect_anomalies(df)

    assert len(result) == len(df)


def test_anomaly_detection_rejects_invalid_contamination():
    df = pd.DataFrame({"Value": [1, 2, 3]})

    with pytest.raises(ValueError):
        detect_anomalies(df, contamination=0)


def test_anomaly_detection_without_numeric_columns():
    df = pd.DataFrame({
        "Name": ["Aarav", "Priya", "Rahul"],
    })

    result = detect_anomalies(df)

    assert result.empty
