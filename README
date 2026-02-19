# CW1 – Machine Learning Pipeline

**Module:** 5CCSAMLF – Machine Learning  
**Student ID:** K23114605  

## Overview

A regression pipeline to predict diamond `outcome` from 30 features. The final model is a **Gradient Boosting Regressor** selected via 5-fold cross-validation (R² = 0.4719).

## Repository Structure

```
cw1/
├── data/                → CW1_train.csv, CW1_test.csv
├── evaluate/            → CW1_eval_script.py
├── figures/             → EDA visualisations
├── outputs/             → CW1_submission_K23114605.csv
├── report/              → report.pdf
├── src/
│   ├── eda_cleaning.py  → EDA and data cleaning plots
│   └── train_model.py   → Final submission script (clean + train + predict)
└── train/
    └── cw.py            → Full exploration: model comparison, CV, hyperparameter tuning
```

## How to Run

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib xgboost joblib

# Generate EDA figures
python src/eda_cleaning.py

# Train final model and produce submission CSV
python src/train_model.py
```

The submission file will be saved to `outputs/CW1_submission_K23114605.csv`.

## Pipeline Summary

1. **Data Cleaning** – Remove physically impossible dimensions (x, y, z = 0) and extreme outliers in depth, table, and dimensions.
2. **Preprocessing** – Median imputation + standardisation for numerical features; one-hot encoding for categorical features (cut, color, clarity). All steps inside an sklearn Pipeline to prevent data leakage.
3. **Model Selection** – Compared Ridge Regression, Random Forest, XGBoost, and Gradient Boosting via 5-fold CV. GBR achieved the highest mean R².
4. **Hyperparameter Tuning** – RandomizedSearchCV (20 iterations, 5-fold CV) over learning rate, number of estimators, max depth, and subsample rate.
5. **Prediction** – Final model retrained on all cleaned training data and used to predict 1,000 test samples.
