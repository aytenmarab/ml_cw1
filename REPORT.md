## CW1 Modelling Report (updated 2026-02-17)

### Data & Cleaning
- Source files: `CW1_train.csv` (10,000 rows, 31 columns) and `CW1_test.csv` (1,000 rows).
- Cleaning rules applied before modelling: drop rows with non-positive physical dimensions (`x`, `y`, `z`); keep sensible ranges `45 < depth < 75`, `40 < table < 80`, `x,y,z < 30`. Only 5 rows are removed → final training size 9,995.
- No missing values found in the raw data.
- Outcome stats after cleaning: mean −4.98, std 12.72; quartiles (25/50/75%) = −13.99, −5.44, 3.91.

### Quick EDA highlights
- Categorical distributions (%):  
  - cut: Ideal 40.4, Premium 24.4, Very Good 23.0, Good 9.3, Fair 3.0  
  - color: G 21.2, E 18.7, F 17.5, H 15.1, D 12.5, I 9.8, J 5.3  
  - clarity: SI1 24.1, VS2 22.6, SI2 17.4, VS1 15.0, VVS2 9.5, VVS1 6.8, IF 3.2, I1 1.5
- Numeric correlations with the target (top absolute):  
  - depth strongly negative (≈ −0.41).  
  - b3, b1, a1, a4, table show the strongest positive correlations (0.11–0.23).  
  - carat and price have near-zero linear correlation with the provided target.

### Modelling pipeline
- Preprocessing: median-impute and standardise numeric features; most-frequent impute + one-hot encode categoricals (`cut`, `color`, `clarity`); all wrapped in a `ColumnTransformer`.
- Candidates (cross-validation comparison in the script): GradientBoostingRegressor, RandomForestRegressor, Ridge, and optional XGBoost if installed.
- Previous 5-fold CV (before refactor) for reference: GBR ≈0.472 ±0.010, RF ≈0.456 ±0.010, XGB ≈0.462 ±0.014, Ridge ≈0.285 ±0.018 (R²). GBR was the winner.

### Hyperparameter tuning plan
- Added `--tune` flag to run a `RandomizedSearchCV` on GBR (3-fold). Search space:  
  `n_estimators ∈ [200,1200]`, `learning_rate 0.01–0.10`, `max_depth 2–5`, `min_samples_leaf 1–7`, `subsample 0.6–1.0`, `max_features ∈ {sqrt, None}`.
- Use `python train_and_submit.py --fast --tune` for a quicker run (no model comparison first). Expect several minutes; best params and CV R² will be printed for reporting.

### How to run now
- Fast path (no CV, trains default GBR and writes submission/model):  
  `python train_and_submit.py --fast`
- Full comparison (3-fold CV across candidates; slower):  
  `python train_and_submit.py`
- Output files: model `final_model.pkl`, submission `CW1_submission_K23114605.csv`.

### Notes for the written report
- Explain cleaning rationale (only 5 rows dropped → minimal data loss, removes physically impossible gems and extreme measurements).  
  Link observed target–depth negative relationship to domain intuition: overly deep stones may be valued lower relative to calibration target.  
  Mention that categorical imbalance is mild; one-hot encoding is appropriate.  
  Include tuned GBR parameters and CV score once you run `--tune`.  
  State evaluation metric: R² via cross-validation; final model refit on full cleaned data before test prediction.
