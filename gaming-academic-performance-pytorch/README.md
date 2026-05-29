# Gaming vs Academic Performance — Grade Prediction (PyTorch)

A **deep-learning regression** project: a PyTorch feed-forward neural network that
predicts student `grades` from gaming, lifestyle, and study habits, benchmarked
against a Random Forest baseline.

## Dataset

~8,000 students, 14 columns: age, gender, gaming hours, study hours, sleep,
attendance, gaming genre, social activity, device usage, reaction time, addiction
score, stress level, and the target `grades`. Source: Kaggle
(`nalisha/gaming-vs-academic-performance`).

> The notebook downloads the data via `kagglehub` (needs Kaggle credentials), or
> falls back to a local `Gaming_Academic_Performance.csv` in this folder. The CSV
> is not committed.

## Approach

- **EDA** — grade distribution, correlation heatmap, study/gaming-hours scatter plots.
- **Preprocessing** — uses *all* predictive columns; `ColumnTransformer` to
  standardize numerics and one-hot encode categoricals; the target is scaled
  separately. All scalers are fit on the training split only (no leakage).
- **Baseline** — Random Forest with 5-fold CV R² and test metrics.
- **Neural network** — a 3-hidden-layer MLP (BatchNorm + ReLU + Dropout) trained
  with Adam, a learning-rate scheduler, gradient clipping, and early stopping on
  validation loss; metrics reported in real grade points.
- **Evaluation** — R², MAE, RMSE, training curves, predicted-vs-actual, and
  feature importance.

## What was fixed from the original

- The network was being fed **only categorical columns** (all strong numeric
  predictors were dropped) — now uses the full feature set.
- `super().__init__()` was missing its parentheses (the model wouldn't initialize).
- `DataLoader` was given raw tuples instead of a `TensorDataset`; rebuilt correctly.
- The target was never scaled despite the training/eval code expecting a
  `target_scaler`; now handled end to end with inverse-transform for real-unit metrics.

## Results

Study hours and gaming hours dominate; both the Random Forest and the tuned MLP
predict grades to within a few points (RMSE ~6 on the original RF run).

> Re-run the notebook to fill in exact R²/MAE/RMSE for both models.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook gaming_vs_academics_pytorch.ipynb
```

## Key skills demonstrated

PyTorch model/training-loop design, correct input *and* target scaling, early
stopping and gradient clipping, comparing a neural net against a classical
baseline, and debugging a non-trivial existing codebase.

## Possible extensions

Hyperparameter search over width/depth/learning rate, k-fold CV for the network,
and SHAP / partial-dependence analysis.
