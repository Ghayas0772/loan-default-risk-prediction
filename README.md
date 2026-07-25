# loan-default-risk-prediction
Predictive machine learning models to assess consumer loan default risk, built on a dataset of 190,000+ lending records. Achieved an ROC-AUC of 0.77 using optimized XGBoost and CatBoost models through exhaustive feature engineering and selection.

##Project Overview

This project walks through the full ML lifecycle for a binary classification problem — predicting whether a consumer loan will default — from raw data audit through final model evaluation.

##Notebooks
**Notebook**	**Description**
```
01_data_audit_and_understanding.ipynb	Initial data audit — schema review, data quality checks, exploratory analysis
02_accepted_data_cleaning_milestone2.ipynb	Data cleaning, handling missing values, outlier treatment
03_[add your 3rd notebook name].ipynb	[Add description — e.g., feature engineering / feature selection]
04_modeling_ca_loan_default.ipynb	Model training, tuning, and evaluation (XGBoost, CatBoost)
```
Key Results
ROC-AUC: 0.77 on held-out test data
Models: XGBoost, CatBoost
Dataset size: 190,000+ lending records
Techniques: feature engineering, feature selection, hyperparameter tuning
Tech Stack
Python (pandas, NumPy, scikit-learn)
XGBoost, CatBoost
Google Colab / Jupyter Notebook
Setup
bash
pip install -r requirements.txt
Data

[Add a note here on where the dataset came from and how to obtain it, since raw data is not included in this repo — e.g., "Dataset sourced from [source]. Download instructions: ..."]

Author

Ghayasudin Ghayas — LinkedIn | GitHub
