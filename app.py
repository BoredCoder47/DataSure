
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset, get_dataset_summary
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
from modules.anomaly_detection import detect_anomalies
from modules.quality_score import (
    calculate_quality_score,
    get_quality_rating,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Dataset Quality Checker",
    page_icon="🔍",
    layout="wide",
)

st.title("AI Dataset Quality Checker")
st.write(
    "Upload a dataset to inspect its structure, identify "
    "data-quality problems, calculate a quality score, "
    "and detect unusual records using machine learning."
)


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)


if uploaded_file is not None:
    try:
        # ------------------------------------------
        # LOAD DATASET
        # ------------------------------------------

        df = load_dataset(uploaded_file)

        st.success("Dataset loaded successfully.")

        # ------------------------------------------
        # DATASET OVERVIEW
        # ------------------------------------------

        st.subheader("Dataset Overview")

        summary = get_dataset_summary(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", summary["rows"])
        col2.metric("Columns", summary["columns"])
        col3.metric("Missing Cells", summary["missing_cells"])
        col4.metric("Duplicate Rows", summary["duplicate_rows"])

        # ------------------------------------------
        # RUN QUALITY CHECKS
        # ------------------------------------------

        missing = detect_missing_values(df)
        duplicates = detect_duplicate_rows(df)
        outliers = detect_iqr_outliers(df)

        category_issues = detect_inconsistent_categories(df)
        date_issues = detect_invalid_dates(df)

        # Default range validation for age columns.

        age_columns = [
            column
            for column in df.columns
            if str(column).strip().casefold() in {"age", "ages"}
        ]

        numeric_rules = {
            column: {"min": 0, "max": 120}
            for column in age_columns
        }

        range_issues = validate_numeric_ranges(
            df,
            numeric_rules,
        )

        # ------------------------------------------
        # CALCULATE DATASET QUALITY SCORE
        # ------------------------------------------

        quality_result = calculate_quality_score(
            df=df,
            category_issues=category_issues,
            range_issues=range_issues,
            date_issues=date_issues,
        )

        overall_score = quality_result["overall_score"]
        dimension_scores = quality_result["dimensions"]
        quality_rating = get_quality_rating(overall_score)

        # ------------------------------------------
        # DATASET QUALITY SCORE DISPLAY
        # ------------------------------------------

        st.divider()
        st.header("Dataset Quality Score")

        score_col, rating_col = st.columns([1, 2])

        with score_col:
            st.metric(
                "Overall Quality",
                f"{overall_score:.2f}/100",
            )

        with rating_col:
            st.subheader(quality_rating)

            st.progress(
                int(round(overall_score)),
                text=f"{overall_score:.2f}% quality score",
            )

        st.caption(
            "This is a project-defined heuristic, not a universal "
            "data-quality standard. Review the individual dimensions "
            "before interpreting the overall score."
        )

        st.write("**Quality Dimension Breakdown**")

        dimension_labels = {
            "completeness": "Completeness",
            "uniqueness": "Uniqueness",
            "validity": "Validity",
            "consistency": "Consistency",
        }

        dimension_columns = st.columns(4)

        for column, (key, label) in zip(
            dimension_columns,
            dimension_labels.items(),
        ):
            value = dimension_scores[key]

            with column:
                st.metric(label, f"{value:.2f}/100")
                st.progress(int(round(value)))

        with st.expander("How is the score calculated?"):
            st.markdown(
                """
                | Dimension | Weight | Measurement |
                |---|---:|---|
                | Completeness | 30% | Proportion of non-missing cells |
                | Uniqueness | 25% | Proportion of non-duplicate rows |
                | Validity | 25% | Proportion of cells without detected range/date violations |
                | Consistency | 20% | Category values not involved in detected formatting inconsistencies |

                **Overall score:** weighted sum of the four dimension scores.

                IQR outliers and Isolation Forest anomalies do not directly
                reduce the score because unusual values are not necessarily errors.
                """
            )

        with st.expander("Quality issue counts"):
            st.json(quality_result["issues"])

        # ------------------------------------------
        # DATASET PREVIEW
        # ------------------------------------------

        st.divider()
        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            width="stretch",
        )

        # ------------------------------------------
        # COLUMN INFORMATION
        # ------------------------------------------

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values,
        })

        st.dataframe(
            column_info,
            width="stretch",
            hide_index=True,
        )

        # ------------------------------------------
        # QUALITY ANALYSIS TABS
        # ------------------------------------------

        st.divider()
        st.header("Quality Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Missing Values",
            "Duplicate Rows",
            "Outliers (IQR)",
            "Data Validation",
            "AI Anomaly Detection",
        ])

        # ==========================================
        # TAB 1: MISSING VALUES
        # ==========================================

        with tab1:
            st.subheader("Missing Values")

            st.write(
                "Shows the number and percentage of missing "
                "values in each column."
            )

            missing_issues = missing[
                missing["missing_count"] > 0
            ]

            if missing_issues.empty:
                st.success("No missing values found.")
            else:
                st.warning(
                    f"Found missing values in "
                    f"{len(missing_issues)} column(s)."
                )

                st.dataframe(
                    missing_issues,
                    width="stretch",
                    hide_index=True,
                )

        # ==========================================
        # TAB 2: DUPLICATE ROWS
        # ==========================================

        with tab2:
            st.subheader("Duplicate Rows")

            st.write(
                "Shows repeated records, excluding the first "
                "occurrence of each row."
            )

            if duplicates.empty:
                st.success("No duplicate rows found.")
            else:
                st.warning(
                    f"Found {len(duplicates)} duplicate row(s)."
                )

                st.dataframe(
                    duplicates,
                    width="stretch",
                )

        # ==========================================
        # TAB 3: IQR OUTLIERS
        # ==========================================

        with tab3:
            st.subheader("Statistical Outlier Detection")

            st.write(
                "The Interquartile Range (IQR) method flags "
                "unusually low or high numerical values. "
                "Flagged values are not necessarily incorrect."
            )

            found_outliers = False

            for column, items in outliers.items():
                if items:
                    found_outliers = True

                    st.write(f"**Column: {column}**")

                    st.dataframe(
                        pd.DataFrame(items),
                        width="stretch",
                        hide_index=True,
                    )

            if not found_outliers:
                st.success("No IQR outliers found.")

        # ==========================================
        # TAB 4: DATA VALIDATION
        # ==========================================

        with tab4:
            st.subheader("Inconsistent Categories")

            st.write(
                "Identifies category values that differ only "
                "by capitalization or surrounding whitespace."
            )

            if category_issues.empty:
                st.success(
                    "No capitalization or whitespace "
                    "inconsistencies found."
                )
            else:
                st.warning(
                    f"Found {len(category_issues)} "
                    "inconsistent category group(s)."
                )

                st.dataframe(
                    category_issues,
                    width="stretch",
                    hide_index=True,
                )

            st.divider()
            st.subheader("Numeric Range Validation")

            st.write(
                "Age and Ages columns are checked against "
                "the default range of 0 to 120."
            )

            if not age_columns:
                st.info(
                    "No Age or Ages column was detected. "
                    "No default numeric range rules were applied."
                )
            elif range_issues.empty:
                st.success(
                    "No age values violate the configured range."
                )
            else:
                st.warning(
                    f"Found {len(range_issues)} "
                    "numeric range violation(s)."
                )

                st.dataframe(
                    range_issues,
                    width="stretch",
                    hide_index=True,
                )

            st.divider()
            st.subheader("Date Validation")

            st.write(
                "Checks columns whose names contain date, time, "
                "or dob for non-empty values that cannot be parsed."
            )

            if date_issues.empty:
                st.success(
                    "No invalid dates found in detected date columns."
                )
            else:
                st.warning(
                    f"Found {len(date_issues)} invalid date value(s)."
                )

                st.dataframe(
                    date_issues,
                    width="stretch",
                    hide_index=True,
                )

        # ==========================================
        # TAB 5: AI ANOMALY DETECTION
        # ==========================================

        with tab5:
            st.subheader("Isolation Forest Anomaly Detection")

            st.write(
                "Isolation Forest identifies numerical records "
                "that differ from the patterns in your dataset. "
                "Flagged records are candidates for review, "
                "not guaranteed errors."
            )

            contamination = st.slider(
                "Expected anomaly proportion",
                min_value=0.01,
                max_value=0.30,
                value=0.10,
                step=0.01,
                help=(
                    "Controls the approximate proportion of records "
                    "the model flags. This is a model setting, "
                    "not a measured error rate."
                ),
            )

            anomalies = detect_anomalies(
                df,
                contamination=contamination,
            )

            if anomalies.empty:
                st.warning(
                    "Anomaly detection requires at least two rows "
                    "and at least one usable numerical column."
                )
            else:
                flagged = anomalies[
                    anomalies["is_anomaly"]
                ]

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Flagged Records",
                    len(flagged),
                )

                metric2.metric(
                    "Records Analyzed",
                    len(anomalies),
                )

                st.caption(
                    "Higher anomaly scores indicate more unusual "
                    "records. Scores are not probabilities."
                )

                st.subheader("Anomaly Results")

                st.dataframe(
                    anomalies,
                    width="stretch",
                    hide_index=True,
                )

                if not flagged.empty:
                    st.subheader("Flagged Record Details")

                    flagged_indices = flagged[
                        "row_index"
                    ].tolist()

                    st.dataframe(
                        df.loc[flagged_indices],
                        width="stretch",
                    )
                else:
                    st.success(
                        "No records were flagged by the model."
                    )

    except Exception as exc:
        st.error(f"Could not analyze the dataset: {exc}")


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

else:
    st.info("Upload a CSV or Excel file to begin.")

    st.markdown(
        """
        **Available features**

        - Dataset preview and column statistics
        - Missing-value detection
        - Duplicate-row detection
        - Statistical outlier detection using IQR
        - Inconsistent category detection
        - Numeric range validation for age columns
        - Invalid date detection
        - ML-based anomaly detection using Isolation Forest
        - Weighted dataset quality score out of 100
        """
    )
