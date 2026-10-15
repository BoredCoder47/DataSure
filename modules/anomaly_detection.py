
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect unusual rows using Isolation Forest.

    Parameters
    ----------
    df:
        Input dataset.
    contamination:
        Expected proportion of anomalies, between 0 and 0.5.
        This is a model setting, not a measured error rate.
    random_state:
        Seed for reproducible results.

    Returns
    -------
    DataFrame with original row index, anomaly flag,
    and anomaly score. Higher scores indicate more unusual rows.
    """

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "contamination must be greater than 0 and at most 0.5."
        )

    result_columns = [
        "row_index",
        "is_anomaly",
        "anomaly_score",
    ]

    if df.empty:
        return pd.DataFrame(columns=result_columns)

    # Isolation Forest needs numerical features.
    numeric_df = df.select_dtypes(include="number").copy()

    if numeric_df.empty:
        return pd.DataFrame(columns=result_columns)

    # Replace infinite values and discard entirely empty columns.
    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        return pd.DataFrame(columns=result_columns)

    # Fill missing numeric values using each column's median.
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(numeric_df)

    if values.shape[1] == 0 or values.shape[0] < 2:
        return pd.DataFrame(columns=result_columns)

    # Isolation Forest labels normal records as 1 and
    # anomalies as -1.
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )

    predictions = model.fit_predict(values)

    # decision_function: lower scores are more anomalous.
    # Reverse the sign so higher scores mean more unusual.
    scores = -model.decision_function(values)

    results = pd.DataFrame({
        "row_index": df.index,
        "is_anomaly": predictions == -1,
        "anomaly_score": scores,
    })

    return results.sort_values(
        "anomaly_score",
        ascending=False,
    ).reset_index(drop=True)
