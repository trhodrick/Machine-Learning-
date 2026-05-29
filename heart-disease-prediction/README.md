# Heart Disease Prediction

A **binary classification** project predicting the presence of heart disease from
clinical measurements, built as a clean, leakage-free workflow.

## Dataset

The UCI/Kaggle heart dataset: 1,025 rows, 13 clinical features (age, sex,
chest-pain type, blood pressure, cholesterol, max heart rate, ST depression, etc.)
and a binary `target`.

> **Important:** 723 of the 1,025 rows are exact duplicates. They are dropped
> first (leaving 302 unique patients) — otherwise identical patients leak across
> the train/test split and inflate the score.

> The CSV (`heart.csv`) is not committed. Place it in the project root before running.

## Approach

- **Data quality** — drop duplicate rows; confirm no missing values.
- **EDA** — target balance, age/heart-rate distributions by outcome, correlations.
- **Preprocessing** — clinical *codes* (chest-pain type, ECG, slope, vessels,
  thalassemia) one-hot encoded; continuous measurements standardized; all inside a
  `Pipeline` fit on training folds only.
- **Modelling** — Logistic Regression, KNN, SVM, Random Forest, AdaBoost compared
  with 5-fold cross-validated ROC AUC; top model tuned via `RandomizedSearchCV`.
- **Evaluation** — accuracy, ROC AUC/curve, confusion matrix, classification
  report (with the correct `(y_true, y_pred)` argument order), feature importance.

## Results

Tuned Random Forest typically reaches ~0.85+ accuracy / ~0.90+ ROC AUC on the
held-out set. Chest-pain type, number of major vessels (`ca`), and max heart rate
are the strongest predictors.

> Re-run the notebook to fill in exact numbers and plots.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

## Key skills demonstrated

Spotting and removing leakage from duplicate rows, correct categorical encoding of
coded clinical variables, pipelines, cross-validated model selection, tuning, and
careful metric reporting.

## Possible extensions

Probability calibration and SHAP explanations for individual patients.
