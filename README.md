# AI Dataset Quality Checker

A Python-based data quality analysis tool built with Streamlit, Pandas, and Scikit-learn. It helps users identify data-quality issues in CSV and Excel datasets, validate values against configurable rules, detect unusual records using machine learning, and export analysis results.

## Overview

Poor-quality data can lead to unreliable analysis, misleading visualizations, and incorrect machine learning predictions. The AI Dataset Quality Checker provides a single dashboard to inspect datasets, identify common quality issues, calculate a quality score, and generate downloadable reports.

The application combines rule-based validation with statistical techniques and machine learning to help users understand the condition of their data before using it for further analysis.

## Features

### Dataset Analysis
- Upload CSV and Excel files (`.csv`, `.xlsx`, `.xls`).
- Preview dataset records.
- View row and column counts.
- Inspect column data types, missing values, and unique values.
- View dataset memory usage and other summary statistics where provided by the summary module.

### Data Quality Checks
- **Missing-value detection:** Identify columns containing missing values.
- **Duplicate detection:** Find repeated records.
- **IQR outlier detection:** Identify statistically unusual numerical values using the Interquartile Range method.
- **Category consistency:** Detect categorical values that differ in capitalization or surrounding whitespace.
- **Numeric range validation:** Define custom minimum and maximum values for selected columns.
- **Date validation:** Identify unparseable values in columns recognized as date-related.
- **Data-type validation:** Check values against expected types such as Text, Number, Date, and Boolean.

### Machine Learning Anomaly Detection
Uses the Isolation Forest algorithm to identify numerical records that differ from the general patterns in the dataset.

The anomaly detection interface allows users to adjust the expected anomaly proportion. Flagged records are candidates for investigation, not necessarily incorrect records.

### Dataset Quality Score
Calculates an overall quality score out of 100 using four dimensions:

| Dimension | Weight |
|---|---:|
| Completeness | 30% |
| Uniqueness | 25% |
| Validity | 25% |
| Consistency | 20% |

The dashboard also displays individual dimension scores and a quality rating.

The score is a project-defined heuristic, not a universal data-quality standard. Data-type mismatches and statistical or machine-learning anomalies are reported separately and do not directly reduce the score under the current implementation.

### Downloadable Reports
- Export a complete Excel workbook containing separate worksheets for the quality summary, column information, and detected issues.
- Download individual issue reports as CSV files.
- Export the quality summary separately.
- Review configured numeric range rules and quality dimension scores in the Excel report.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Isolation Forest anomaly detection |
| OpenPyXL | Excel report generation |
| Pytest | Automated testing |

## Project Structure

```text
Dataset Checker/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── quality_checks.py
│   ├── data_validation.py
│   ├── anomaly_detection.py
│   └── quality_score.py
│
├── tests/
│   ├── test_quality_checks.py
│   ├── test_anomaly_detection.py
│   └── test_quality_score.py
│
└── sample_data/
    └── sample.csv
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- pip package manager.
- Git, if you want to clone the repository.

### 1. Clone the repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd "Dataset Checker"
```

Alternatively, download the project and open its root directory in CMD.

### 2. Create a virtual environment

**Windows CMD:**

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ensure that `requirements.txt` includes all dependencies used by the project, including `streamlit`, `pandas`, `numpy`, `scikit-learn`, `openpyxl`, and `pytest` if you use the included tests.

### 4. Run the application

```cmd
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## How to Use

1. Launch the application.
2. Upload a CSV or Excel dataset.
3. Review the dataset overview and column information.
4. Configure expected data types for columns where necessary.
5. Select numeric columns and configure acceptable value ranges.
6. Review missing values, duplicate records, and IQR outliers.
7. Inspect data-type mismatches, inconsistent categories, numeric range violations, and invalid dates.
8. Adjust the Isolation Forest contamination parameter and inspect flagged records.
9. Review the overall quality score and its individual dimensions.
10. Download the complete Excel report or individual CSV reports.

## Algorithms and Methodology

### 1. Missing-Value Analysis

Identifies missing entries and summarizes their occurrence across columns.

### 2. Duplicate Detection

Compares dataset records to identify repeated rows.

### 3. Interquartile Range (IQR)

The IQR method uses the first quartile \(Q_1\) and third quartile \(Q_3\):

\[
IQR = Q_3 - Q_1
\]

The conventional lower and upper boundaries are:

\[
L = Q_1 - 1.5(IQR)
\]

\[
U = Q_3 + 1.5(IQR)
\]

Values outside these boundaries are flagged as potential outliers.

### 4. Rule-Based Validation

Checks values against expected data types and user-configured numeric ranges. It also identifies certain categorical formatting inconsistencies and invalid dates in detected date-related columns.

### 5. Isolation Forest

Isolation Forest identifies unusual numerical records by isolating observations in randomly constructed trees. Observations that are easier to isolate are considered more anomalous.

The contamination parameter controls the expected proportion of observations to flag. It should not be interpreted as the actual error rate.

### 6. Quality Score

The overall score is a weighted combination of completeness, uniqueness, validity, and consistency scores.

The rating thresholds are:

| Score | Rating |
|---|---|
| 90–100 | Excellent |
| 75–89.99 | Good |
| 60–74.99 | Fair |
| Below 60 | Poor |

These ratings are specific to this implementation and should be interpreted alongside the detailed issue reports.

## Running Tests

Run the automated tests from the project root:

```cmd
python -m pytest -v
```

The tests cover the quality-checking, anomaly-detection, and quality-scoring modules.

Review the test output to confirm that the current version passes. Additional tests for file uploads, validation edge cases, and report exports can further improve reliability.

## Limitations

- The quality score uses project-defined rules and weights; different applications may require different scoring criteria.
- IQR outliers and Isolation Forest anomalies are indicators for investigation, not proof of incorrect data.
- Automatic date detection depends on the date-related column names and the behavior implemented in the validation module.
- The anomaly detection model focuses on numerical features and does not independently understand the semantic meaning of a record.
- Data-type mismatches are reported separately and do not directly affect the current quality score.
- The application currently focuses on dataset inspection and reporting; it does not automatically clean or overwrite the uploaded dataset.
- Very large datasets may require additional performance optimization and memory management.

## Future Enhancements

Potential improvements include:

- Safe data cleaning with preview and before-and-after comparisons.
- Interactive visualizations for distributions, correlations, and missing-value patterns.
- Saving and reusing validation rules across datasets.
- More comprehensive edge-case and integration testing.
- Additional validation rules for domain-specific datasets.
- Improved report styling and more detailed audit information.

## Privacy and Data Handling

The application is designed to process uploaded datasets within the running application. When deploying it to a hosted environment, review the hosting configuration, access controls, logging, and data-retention behavior before uploading sensitive information.

Do not upload confidential or personally identifiable information to an untrusted deployment.

*Built with Python, Streamlit, Pandas, and Scikit-learn.*
