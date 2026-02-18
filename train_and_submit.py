#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
import joblib

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent
TRAIN_PATH = DATA_DIR / "CW1_train.csv"
TEST_PATH = DATA_DIR / "CW1_test.csv"
SUB_PATH = DATA_DIR / "CW1_submission_K23114605.csv"
MODEL_PATH = DATA_DIR / "final_model.pkl"

# Load the data
trn = pd.read_csv(TRAIN_PATH)  # 10000 rows with the outcome
tst = pd.read_csv(TEST_PATH)   # 1000 rows without outcome


# EDA GRAPHS (BEFORE CLEANING)
# Graph 1: distribution of target variable (before cleaning)
plt.figure()
plt.hist(trn["outcome"], bins=50)
plt.xlabel("Outcome value")
plt.ylabel("Frequency")
plt.title("Distribution of target variable (before cleaning)")
plt.show()

# Graph 2: physical dimensions x vs y (shows invalid zeros / unrealistic points)
plt.figure()
plt.scatter(trn["x"], trn["y"], alpha=0.3)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Diamond dimensions (x vs y) before cleaning")
plt.show()

# Graph 3: depth outliers (before cleaning)
plt.figure()
plt.boxplot(trn["depth"], vert=False)
plt.xlabel("Depth")
plt.title("Boxplot of depth before cleaning")
plt.show()


# Data cleaning
# remove dimensionless diamonds, x, y, z are the diamond's physical dimensions in millimetres
trn = trn[(trn["x"] > 0) & (trn["y"] > 0) & (trn["z"] > 0)]

# remove outliers in physical measurements
trn = trn[(trn["depth"] > 45) & (trn["depth"] < 75)]
trn = trn[(trn["table"] > 40) & (trn["table"] < 80)]
trn = trn[(trn["x"] < 30)]
trn = trn[(trn["y"] < 30)]
trn = trn[(trn["z"] < 30)]

print(f"Training samples after cleaning: {len(trn)}")

#confirmation plot for depth
plt.figure()
plt.boxplot(trn["depth"], vert=False)
plt.xlabel("Depth")
plt.title("Boxplot of depth after cleaning")
plt.show()

y = trn["outcome"]  # target variable what were trying to predict
X = trn.drop(columns=["outcome"])  # input features

categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = []
for c in X.columns:
    if c not in categorical_cols:
        numerical_cols.append(c)

# preprocessing pipelines
# Numerical preprocessing: median impute then standardise
numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # fill missing values with median
    ("scaler", StandardScaler())
])

# when fitted: missing numeric values are replaced by median and values are scaled
# Categorical preprocessing: most-frequent impute then one-hot encode
categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing with most common
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer([
    ("numeric", numeric_tf, numerical_cols),
    ("categorical", categorical_tf, categorical_cols)
])

# XGBoost as a candidate + tuned hyperparameters
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print("XGBoost not installed — skipping. pip install xgboost to enable.")

# Candidate models (baseline comparison)
candidates = {
    "gbr": GradientBoostingRegressor(
        random_state=42, n_estimators=500, learning_rate=0.03, max_depth=3
    ),
    "rf": RandomForestRegressor(
        random_state=42, n_estimators=400, max_depth=None,
        min_samples_leaf=1, n_jobs=-1
    ),
    "ridge": Ridge(alpha=5.0),
}

if xgb_available:
    candidates["xgb"] = XGBRegressor(
        random_state=42, n_estimators=600, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1
    )

# cross validation
r2 = make_scorer(r2_score)
cv_result = {}

for name, model in candidates.items():
    pipe = Pipeline([("pre", preprocessing), ("model", model)])
    scores = cross_val_score(
        pipe, X, y,
        cv=5,
        scoring=r2,
        n_jobs=-1
    )
    cv_result[name] = (scores.mean(), scores.std())
    print(f"{name}: R^2 mean = {scores.mean():.4f} std = {scores.std():.4f}")

# selecting the best model
# pick best by mean CV R^2
best_name = max(cv_result, key=lambda k: cv_result[k][0])
best_model = candidates[best_name]
print(f"\nBest model: {best_name}")

# Proper GBR hyperparameter tuning (only if GBR is the best)
# IMPORTANT: we tune the FULL pipeline (preprocessing + model) so tuning matches final training
if best_name == "gbr":
    base_pipe = Pipeline([
        ("pre", preprocessing),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    # Hyperparameter search space (for the GBR inside the pipeline)
    param_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.6, 0.8, 1.0]
    }

    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    # Run search on ALL cleaned training data (CV is inside RandomizedSearchCV)
    random_search.fit(X, y)

    # Best model after tuning (this is already fitted)
    final_pipe = random_search.best_estimator_

    print("Best tuned GBR parameters:", random_search.best_params_)
    print("Best tuned cross-validated R²:", random_search.best_score_)

else:
    # train final model on all cleaned training data (no tuning for other models)
    final_pipe = Pipeline([("pre", preprocessing), ("model", best_model)])
    final_pipe.fit(X, y)

# save model
joblib.dump(final_pipe, MODEL_PATH)

# Predict test set and save submission
yhat = final_pipe.predict(tst)  # produces 1000 predictions
assert yhat.shape[0] == len(tst)
out = pd.DataFrame({"yhat": yhat.astype(float)})
out.to_csv(SUB_PATH, index=False)
print(f"Saved submission to {SUB_PATH}")
