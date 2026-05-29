# Diabetes Disease-Progression Regression

A compact, complete **regression** project predicting diabetes disease progression
one year after baseline, using the classic scikit-learn diabetes dataset.

## Dataset

442 patients, 10 baseline features (already mean-centered and scaled): age, sex,
BMI, blood pressure, and six blood-serum measurements. Loaded directly from
`sklearn.datasets.load_diabetes` — no download required.

## Approach

- **EDA** — target distribution and feature/target correlations.
- **Model comparison** — Linear Regression, Ridge, Lasso, Random Forest, Gradient
  Boosting, compared with 5-fold cross-validated R² in pipelines.
- **Tuning** — `RandomizedSearchCV` on Gradient Boosting.
- **Evaluation** — R², MAE, RMSE, predicted-vs-actual plot, feature importance.

## Results

This dataset has a real ceiling — ~0.45–0.50 R² is typical and expected. The point
is sound methodology and honest reporting rather than an inflated score. **BMI** and
serum measure **s5** are consistently the strongest predictors.

> Re-run the notebook to fill in exact R²/MAE/RMSE.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook diabetes_progression_regression.ipynb
```

## Key skills demonstrated

Comparing linear and ensemble regressors, cross-validation, hyperparameter tuning,
and honest interpretation of a problem with a genuine performance ceiling.

## Possible extensions

Polynomial/interaction features for the linear models; partial-dependence plots for
BMI and blood pressure.
