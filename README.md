# Loan Default Risk Prediction

Predictive machine learning models to assess consumer loan default risk, built on LendingClub's historical loan dataset. The pipeline covers data auditing, cleaning, California-subset extraction, macroeconomic feature merging (rejected loan volume, unemployment rate), and classification modeling (XGBoost, CatBoost).

## Data Source

- **Source:** LendingClub accepted & rejected loan applications (2007–2018), plus California unemployment rate data (state labor statistics)
- **Original accepted dataset:** 1,369,566 rows × 152 columns
- **Original rejected dataset:** ~27.6 million rows, aggregated to 140 monthly macro records (rejected application volume, avg amount, avg DTI, avg risk score)
- **Target variable:** `target_default` (binary: 0 = no default, 1 = default)
- **California subset:** filtered via `addr_state == "CA"`, merged with rejected CA loans and CA unemployment rate by `year_month`

>  **[FINAL SAMPLE SIZE — CONFIRM BEFORE PUBLISHING]**
> Final modeling dataset size: `[X rows]` — pull this from `X_train.shape` / `X.shape` in notebook 04, right before `.fit()`.

## Notebooks

| Notebook | Milestone | Description |
|---|---|---|
| `01_data_audit_and_understanding.ipynb` | 1 | Data audit — schema review, column classification (ID / Leakage / Predictor / Target) |
| `02_Accepted_Data_Cleaning_Milestone2.ipynb` | 2 | Cleaning accepted loans — dropping ID/leakage columns, missing value handling, one-hot encoding |
| `03_Data_Merging_Milestone3.ipynb` | 3 | California-only extraction, rejected loan aggregation, unemployment rate integration, feature correlation analysis |
| `04_Modeling_CA_Loan_Default.ipynb` | 4 | Model training, tuning, and evaluation (XGBoost, CatBoost) |

## Methodology

### Milestone 1 — Data Audit
- Classified 152 raw columns into ID (6), Leakage (38), Predictor (106), and Target categories

### Milestone 2 — Cleaning & Encoding
- Dropped ID/leakage columns, imputed missing values (median for numeric, mode for categorical), one-hot encoded categoricals

### Milestone 3 — California Subset, Macro Data & Feature Selection
- Filtered to California-only loans
- Aggregated 27.6M raw rejected applications into 140 monthly macro features (rejected volume, avg amount, avg DTI, avg risk score)
- Cleaned and merged California unemployment rate data by `year_month`
- Ran correlation analysis against `target_default`; reduced to top correlated features (correlation threshold > 0.05)

### Milestone 4 — Modeling
- Trained XGBoost and CatBoost classifiers on the final California modeling dataset
- Evaluated via ROC-AUC

## Key Results

> ⚠️ **[CONFIRM BEFORE PUBLISHING]**
> - ROC-AUC: `[confirm final value from notebook 04]`
> - Final training set size: `[confirm]`
> - Models: XGBoost, CatBoost

## Tech Stack

- Python (pandas, NumPy, scikit-learn)
- XGBoost, CatBoost
- Google Colab / Jupyter Notebook, Google Drive

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Data

Raw data not included due to file size and LendingClub's data usage terms. Source data: LendingClub historical accepted/rejected loans (2007–2018), commonly mirrored on Kaggle; California unemployment data from state labor force statistics.

## Author

Ghayasudin Ghayas — [LinkedIn](https://linkedin.com/in/ghayasudin-ghayas) | [GitHub](https://github.com/Ghayas0772)
