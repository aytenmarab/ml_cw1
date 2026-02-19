# train_model.py
# Final submission script
# - Loads training and test data
# - Cleans TRAIN data only
# - Trains the final best model with fixed hyperparameters
# - Generates predictions on the test set
# - Saves submission CSV

import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


# File paths (relative to repo root)
TRAIN_PATH = "data/CW1_train.csv"
TEST_PATH = "data/CW1_test.csv"
SUB_PATH = "outputs/CW1_submission_K23114605.csv"


# Data cleaning (TRAIN ONLY)
def clean_train_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove physically impossible dimensions
    for c in ["x", "y", "z"]:
        if c in df.columns:
            df = df[df[c] > 0]

    # Remove extreme / implausible outliers
    if "depth" in df.columns:
        df = df[(df["depth"] > 45) & (df["depth"] < 75)]

    if "table" in df.columns:
        df = df[(df["table"] > 40) & (df["table"] < 80)]

    for c in ["x", "y", "z"]:
        if c in df.columns:
            df = df[df[c] < 30]

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # If there are no categorical columns, we still create a valid transformer list
    transformers = [("num", num_pipe, num_cols)]

    if len(cat_cols) > 0:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def main():
    # Check input files exist
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Could not find training file: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Could not find test file: {TEST_PATH}")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Clean training data only
    train_df = clean_train_df(train_df)

    if "outcome" not in train_df.columns:
        raise ValueError("Training file must contain the target column 'outcome'.")

    y = train_df["outcome"].astype(float)
    X = train_df.drop(columns=["outcome"])

    # Build preprocessing
    preprocessor = build_preprocessor(X)

    # Final best model (hardcoded hyperparameters)
    final_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=1.0,
        random_state=0
    )

    # Full pipeline
    final_pipe = Pipeline([
        ("pre", preprocessor),
        ("model", final_model),
    ])

    # Train on all cleaned training data
    final_pipe.fit(X, y)

    # Predict test set
    yhat = final_pipe.predict(test_df)

    # Safety check (must output exactly one prediction per test row)
    if len(yhat) != len(test_df):
        raise ValueError(
            f"Prediction row count mismatch: got {len(yhat)}, expected {len(test_df)}"
        )

    # Ensure outputs folder exists
    os.makedirs(os.path.dirname(SUB_PATH), exist_ok=True)

    # Save submission (single column 'yhat')
    submission = pd.DataFrame({"yhat": yhat.astype(float)})
    submission.to_csv(SUB_PATH, index=False)

    print(f"Submission saved to {SUB_PATH}")
    print(f"   Rows: {len(submission)}  Columns: {list(submission.columns)}")


if __name__ == "__main__":
    main()
