# Loan Approval Prediction

End-to-end **binary classification** project predicting whether a loan
application is approved, using applicant financials and credit history. Built to
demonstrate a complete, leakage-free machine-learning workflow across several
model families.

## Dataset

50,000 loan applications with 20 columns (after dropping the `customer_id` key):

- **Numeric:** age, years employed, annual income, credit score, credit history,
  savings, current debt, defaults, delinquencies, derogatory marks, loan amount,
  interest rate, and three income-ratio features.
- **Categorical:** occupation status, product type, loan intent.
- **Target:** `loan_status` (1 = approved, 0 = declined), ~55% approved.

> The CSV (`Loan_approval_data_2025.csv`) is not committed. Place it in the
> project root before running.

## Approach

- **EDA** — target balance, numeric distributions, approval rate by category, and
  a correlation matrix.
- **Preprocessing** — a scikit-learn `ColumnTransformer` (standardize numerics +
  one-hot encode categoricals) wrapped in a `Pipeline` with each estimator, so
  the scaler/encoder are fit on training folds only (no data leakage).
- **Model comparison** — Logistic Regression, RBF SVM, Random Forest, AdaBoost,
  Gradient Boosting, and XGBoost benchmarked with 5-fold cross-validated accuracy.
- **Tuning** — `RandomizedSearchCV` over the XGBoost hyperparameter space.
- **Evaluation** — accuracy, ROC AUC, classification report, confusion matrix,
  and feature importances on a held-out 30% test set.

## Results

| Model | CV accuracy |
|-------|-------------|
| XGBoost (tuned) | **~0.93** |
| Gradient Boosting | ~0.92 |
| AdaBoost | ~0.91 |
| Random Forest | ~0.90 |
| Logistic Regression | ~0.85 |
| RBF SVM | ~0.84 |

Tuned XGBoost reaches roughly **0.93 test accuracy** and **~0.97 ROC AUC**. Credit
score, debt/income ratios, and interest rate are the strongest predictors.

> Re-run the notebook to regenerate exact numbers and plots, then update this table.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Loan_approval_classification.ipynb
```

## Project structure

```
loan-approval-classification/
├── Loan_approval_classification.ipynb   # full analysis
├── requirements.txt
└── README.md
```

## Key skills demonstrated

Pipelines & `ColumnTransformer`, avoiding train/test leakage, cross-validation,
hyperparameter search, comparing model families, and interpreting results with
feature importance.

## Possible extensions

Probability calibration for risk-based pricing, SHAP values for per-applicant
explanations, and decision-threshold tuning to reflect the business cost of false
approvals vs. false declines.
