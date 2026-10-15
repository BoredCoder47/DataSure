import io

import pandas as pd
import streamlit as st

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
    detect_data_type_mismatches,
)
from modules.anomaly_detection import detect_anomalies
from modules.quality_score import (
    calculate_quality_score,
    get_quality_rating,
)


# --------------------------------------------------
# REPORT EXPORT HELPERS
# --------------------------------------------------

def dataframe_to_csv_bytes(dataframe):
    """Convert a DataFrame into downloadable UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def flatten_iqr_outliers(outliers):
    """Convert column-wise IQR outlier results into a DataFrame."""
    records = []

    for column, items in outliers.items():
        for item in items:
            record = {"Column": str(column)}

            if isinstance(item, dict):
                record.update(item)
            else:
                record["Value"] = item

            records.append(record)

    return pd.DataFrame(records)


def with_row_index(dataframe, column_name="Row Index"):
    """Preserve original DataFrame index values in exported reports."""
    result = dataframe.copy()
    result.insert(0, column_name, result.index)
    return result.reset_index(drop=True)


def build_excel_report(
    summary_df,
    column_info,
    missing_df,
    duplicates_df,
    outliers_df,
    type_issues_df,
    category_issues_df,
    range_issues_df,
    date_issues_df,
    anomalies_df,
    quality_dimensions_df,
    range_rules_df,
):
    """Build a multi-sheet Excel quality report in memory."""

    sheets = {
        "Quality Summary": summary_df,
        "Dataset Columns": column_info,
        "Missing Values": missing_df,
        "Duplicate Rows": duplicates_df,
        "IQR Outliers": outliers_df,
        "Type Mismatches": type_issues_df,
        "Category Issues": category_issues_df,
        "Range Violations": range_issues_df,
        "Invalid Dates": date_issues_df,
        "AI Anomalies": anomalies_df,
        "Quality Dimensions": quality_dimensions_df,
        "Range Rules": range_rules_df,
    }

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            export_df = dataframe.copy()

            if export_df.empty and len(export_df.columns) == 0:
                export_df = pd.DataFrame(
                    {"Message": ["No issues found."]}
                )

            export_df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )

            worksheet = writer.sheets[sheet_name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            # Set readable column widths.
            for cells in worksheet.columns:
                column_letter = cells[0].column_letter

                longest = max(
                    len(str(cell.value or ""))
                    for cell in cells
                )

                worksheet.column_dimensions[column_letter].width = min(
                    max(longest + 2, 12),
                    40,
                )

    output.seek(0)
    return output.getvalue()


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

        if df.empty:
            st.warning(
                "The uploaded dataset has no rows. "
                "Please upload a dataset containing records."
            )
            st.stop()

        dataset_key = (
            f"{uploaded_file.name}_"
            f"{getattr(uploaded_file, 'size', 0)}"
        )

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
        # EXPECTED DATA-TYPE CONFIGURATION
        # ------------------------------------------

        st.divider()
        st.header("Configure Expected Data Types")

        st.write(
            "Choose the expected type for each column. "
            "The checker will flag non-empty values that do not "
            "match your selection."
        )

        st.caption(
            "Auto uses automatic detection. Number checks numeric "
            "values, Date checks parseable dates, Boolean accepts "
            "true/false, 1/0, yes/no, and y/n. Text skips "
            "type-mismatch checks for that column."
        )

        expected_types = {}

        supported_types = [
            "Auto",
            "Text",
            "Number",
            "Date",
            "Boolean",
        ]

        with st.expander(
            "Set expected types for columns",
            expanded=True,
        ):
            for column in df.columns:
                with st.container():
                    type_col, sample_col = st.columns([1, 2])

                    with type_col:
                        expected_types[column] = st.selectbox(
                            str(column),
                            options=supported_types,
                            index=0,
                            key=(
                                f"expected_type_"
                                f"{dataset_key}_{column}"
                            ),
                        )

                    with sample_col:
                        examples = (
                            df[column]
                            .dropna()
                            .astype(str)
                            .head(3)
                            .tolist()
                        )

                        st.write("**Sample values**")

                        st.write(
                            ", ".join(examples)
                            if examples
                            else "No non-missing values"
                        )

        # ------------------------------------------
        # CONFIGURABLE NUMERIC RANGE RULES
        # ------------------------------------------

        st.divider()
        st.header("Configure Numeric Range Rules")

        st.write(
            "Select numerical columns and define acceptable "
            "minimum and maximum values. These rules are used "
            "in validation results and the quality score."
        )

        numeric_columns = []
        numeric_series = {}

        for column in df.columns:
            values = df[column]

            non_empty = (
                values.notna()
                & values.astype(str).str.strip().ne("")
            )

            cleaned = (
                values.astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
            )

            converted = pd.to_numeric(
                cleaned.where(non_empty),
                errors="coerce",
            )

            selected_type = expected_types.get(column, "Auto")

            if selected_type in {"Text", "Date", "Boolean"}:
                continue

            explicitly_numeric = selected_type == "Number"

            automatically_numeric = (
                non_empty.any()
                and converted[non_empty].notna().all()
            )

            if explicitly_numeric or automatically_numeric:
                numeric_columns.append(column)
                numeric_series[column] = converted

        age_columns = [
            column
            for column in numeric_columns
            if str(column).strip().casefold() in {"age", "ages"}
        ]

        numeric_rules = {}

        if not numeric_columns:
            st.info(
                "No numeric columns are available. Select Number "
                "for a column above to enable its range settings."
            )

        else:
            type_signature = "|".join(
                f"{column}:{expected_types[column]}"
                for column in df.columns
            )

            selected_columns = st.multiselect(
                "Select columns to validate",
                options=numeric_columns,
                default=age_columns,
                help=(
                    "Number includes explicitly selected numeric "
                    "columns. Auto includes columns whose non-empty "
                    "values are all numeric."
                ),
                key=(
                    f"selected_range_columns_"
                    f"{dataset_key}_{type_signature}"
                ),
            )

            if not selected_columns:
                st.info(
                    "Select at least one column to configure "
                    "numeric range validation."
                )

            for column in selected_columns:
                numeric_values = numeric_series[column].dropna()

                if (
                    str(column).strip().casefold()
                    in {"age", "ages"}
                ):
                    default_minimum = 0.0
                    default_maximum = 120.0

                elif not numeric_values.empty:
                    default_minimum = float(numeric_values.min())
                    default_maximum = float(numeric_values.max())

                else:
                    default_minimum = 0.0
                    default_maximum = 100.0

                with st.expander(
                    f"Range settings: {column}",
                    expanded=True,
                ):
                    min_col, max_col = st.columns(2)

                    with min_col:
                        minimum = st.number_input(
                            f"Minimum value — {column}",
                            value=default_minimum,
                            step=1.0,
                            key=(
                                f"range_min_"
                                f"{dataset_key}_{column}"
                            ),
                        )

                    with max_col:
                        maximum = st.number_input(
                            f"Maximum value — {column}",
                            value=default_maximum,
                            step=1.0,
                            key=(
                                f"range_max_"
                                f"{dataset_key}_{column}"
                            ),
                        )

                    if minimum > maximum:
                        st.error(
                            f"Minimum cannot exceed maximum "
                            f"for {column}."
                        )
                        st.stop()

                    numeric_rules[column] = {
                        "min": minimum,
                        "max": maximum,
                    }

            if numeric_rules:
                st.caption(
                    f"Configured range rules for "
                    f"{len(numeric_rules)} column(s). "
                    "Values equal to the limits are accepted."
                )

        # ------------------------------------------
        # RUN QUALITY CHECKS
        # ------------------------------------------

        missing = detect_missing_values(df)
        duplicates = detect_duplicate_rows(df)
        outliers = detect_iqr_outliers(df)

        category_issues = detect_inconsistent_categories(df)

        range_validation_df = df.copy()

        for column in numeric_rules:
            range_validation_df[column] = numeric_series[column]

        range_issues = validate_numeric_ranges(
            range_validation_df,
            numeric_rules,
        )

        date_issues = detect_invalid_dates(df)

        type_issues = detect_data_type_mismatches(
            df,
            expected_types=expected_types,
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
            "data-quality standard. Review individual dimensions "
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

        for dimension_column, (key, label) in zip(
            dimension_columns,
            dimension_labels.items(),
        ):
            value = dimension_scores[key]

            with dimension_column:
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

                Numeric range rules are configurable. The selected
                minimum and maximum values are used for validation
                and quality scoring.

                Data-type mismatches are reported separately and do
                not currently affect the quality score.

                IQR outliers and Isolation Forest anomalies do not
                directly reduce the score because unusual values
                are not necessarily errors.
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
            "Data Type": [
                str(dtype) for dtype in df.dtypes
            ],
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(
                dropna=True
            ).values,
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
            st.subheader("Data-Type Validation")

            st.write(
                "Checks values against the expected data types "
                "configured above."
            )

            if type_issues.empty:
                st.success(
                    "No data-type mismatches found for the "
                    "configured expected types."
                )
            else:
                st.warning(
                    f"Found {len(type_issues)} "
                    "data-type mismatch(es)."
                )

                st.dataframe(
                    type_issues,
                    width="stretch",
                    hide_index=True,
                )

            st.divider()
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

            if not numeric_rules:
                st.info(
                    "No numeric range rules are configured. "
                    "Select columns and set their limits above."
                )
            elif range_issues.empty:
                st.success(
                    "No values violate the configured numeric ranges."
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
            st.subheader("Active Numeric Range Rules")

            if numeric_rules:
                rules_table = pd.DataFrame([
                    {
                        "Column": column,
                        "Minimum": bounds["min"],
                        "Maximum": bounds["max"],
                    }
                    for column, bounds in numeric_rules.items()
                ])

                st.dataframe(
                    rules_table,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No numeric range rules are active.")

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
                    f"Found {len(date_issues)} "
                    "invalid date value(s)."
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
                key=f"contamination_{dataset_key}",
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

                    valid_indices = [
                        index
                        for index in flagged_indices
                        if index in df.index
                    ]

                    st.dataframe(
                        df.loc[valid_indices],
                        width="stretch",
                    )
                else:
                    st.success(
                        "No records were flagged by the model."
                    )

        # ==========================================
        # DOWNLOADABLE QUALITY REPORTS
        # ==========================================

        st.divider()
        st.header("Download Quality Reports")

        st.write(
            "Export your analysis as an Excel workbook or "
            "download individual issue reports as CSV files."
        )

        # Prepare individual report tables.
        missing_report = missing[
            missing["missing_count"] > 0
        ].copy()

        duplicate_report = with_row_index(duplicates)

        outlier_report = flatten_iqr_outliers(outliers)

        type_report = type_issues.copy()
        category_report = category_issues.copy()
        range_report = range_issues.copy()
        date_report = date_issues.copy()
        anomaly_report = anomalies.copy()

        # Summary worksheet.
        quality_summary = pd.DataFrame([
            {"Metric": "Dataset Name", "Value": uploaded_file.name},
            {"Metric": "Rows", "Value": summary["rows"]},
            {"Metric": "Columns", "Value": summary["columns"]},
            {
                "Metric": "Missing Cells",
                "Value": summary["missing_cells"],
            },
            {
                "Metric": "Duplicate Rows",
                "Value": summary["duplicate_rows"],
            },
            {
                "Metric": "Overall Quality Score",
                "Value": round(float(overall_score), 2),
            },
            {"Metric": "Quality Rating", "Value": quality_rating},
            {
                "Metric": "Missing-Value Columns",
                "Value": len(missing_report),
            },
            {
                "Metric": "Duplicate Records",
                "Value": len(duplicate_report),
            },
            {
                "Metric": "IQR Outlier Values",
                "Value": len(outlier_report),
            },
            {
                "Metric": "Type Mismatches",
                "Value": len(type_report),
            },
            {
                "Metric": "Category Inconsistency Groups",
                "Value": len(category_report),
            },
            {
                "Metric": "Numeric Range Violations",
                "Value": len(range_report),
            },
            {
                "Metric": "Invalid Dates",
                "Value": len(date_report),
            },
            {
                "Metric": "AI-Flagged Records",
                "Value": (
                    int(anomaly_report["is_anomaly"].sum())
                    if (
                        not anomaly_report.empty
                        and "is_anomaly" in anomaly_report.columns
                    )
                    else 0
                ),
            },
        ])

        # Quality dimensions worksheet.
        quality_dimensions = pd.DataFrame([
            {
                "Dimension": dimension_labels[key],
                "Score": round(float(value), 2),
                "Weight": {
                    "completeness": 0.30,
                    "uniqueness": 0.25,
                    "validity": 0.25,
                    "consistency": 0.20,
                }[key],
            }
            for key, value in dimension_scores.items()
        ])

        # Active range rules worksheet.
        range_rules_report = pd.DataFrame([
            {
                "Column": str(column),
                "Minimum": bounds["min"],
                "Maximum": bounds["max"],
            }
            for column, bounds in numeric_rules.items()
        ])

        # Build the Excel workbook.
        excel_bytes = build_excel_report(
            summary_df=quality_summary,
            column_info=column_info,
            missing_df=missing_report,
            duplicates_df=duplicate_report,
            outliers_df=outlier_report,
            type_issues_df=type_report,
            category_issues_df=category_report,
            range_issues_df=range_report,
            date_issues_df=date_report,
            anomalies_df=anomaly_report,
            quality_dimensions_df=quality_dimensions,
            range_rules_df=range_rules_report,
        )

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="Download Complete Excel Report",
                data=excel_bytes,
                file_name="dataset_quality_report.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                width="stretch",
                key=f"excel_report_{dataset_key}",
            )

        with download_col2:
            st.download_button(
                label="Download Quality Summary (CSV)",
                data=dataframe_to_csv_bytes(quality_summary),
                file_name="quality_summary.csv",
                mime="text/csv",
                width="stretch",
                key=f"summary_csv_{dataset_key}",
            )

        # Individual CSV downloads.
        with st.expander("Download individual issue reports"):
            csv_reports = [
                (
                    "Missing Values",
                    missing_report,
                    "missing_values.csv",
                ),
                (
                    "Duplicate Rows",
                    duplicate_report,
                    "duplicate_rows.csv",
                ),
                (
                    "IQR Outliers",
                    outlier_report,
                    "iqr_outliers.csv",
                ),
                (
                    "Type Mismatches",
                    type_report,
                    "type_mismatches.csv",
                ),
                (
                    "Category Issues",
                    category_report,
                    "category_issues.csv",
                ),
                (
                    "Range Violations",
                    range_report,
                    "range_violations.csv",
                ),
                (
                    "Invalid Dates",
                    date_report,
                    "invalid_dates.csv",
                ),
                (
                    "AI Anomalies",
                    anomaly_report,
                    "ai_anomalies.csv",
                ),
            ]

            for label, report_df, filename in csv_reports:
                st.download_button(
                    label=f"Download {label} CSV",
                    data=dataframe_to_csv_bytes(report_df),
                    file_name=filename,
                    mime="text/csv",
                    key=f"csv_{filename}_{dataset_key}",
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
        - Configurable expected data types
        - Inconsistent category detection
        - Configurable numeric range validation
        - Invalid date detection
        - ML-based anomaly detection using Isolation Forest
        - Weighted dataset quality score out of 100
        - Downloadable Excel quality reports
        - Individual CSV issue reports
        """
    )