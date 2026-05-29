# Blood Cell Anomaly Detection

A medical-imaging-adjacent **classification** project on microscopy blood-cell
measurements. Two tasks: detect whether a cell is **normal vs. anomalous**
(binary), and identify its specific **cell type** (19-class). Built to showcase a
careful, leakage-aware machine-learning workflow.

## Dataset

5,880 cells with 36 columns:

- **Cell morphology** — diameter, nucleus area, circularity, eccentricity,
  granularity, lobularity, chromatin density, membrane smoothness, color channels.
- **CBC blood counts** — WBC/RBC/platelet counts, hemoglobin, hematocrit, MCV, MCHC.
- **Patient** — age group, sex.
- **Targets** — `anomaly_label` (0/1, ~32% anomaly) and `cell_type` (19 classes).

> The CSV (`blood_cell_anomaly_detection.csv`) is not committed. Place it in the
> project root before running.

## Approach

- **EDA** — class balance, morphological distributions (normal vs anomaly),
  correlation matrix.
- **Leakage-aware feature selection** — deliberately excludes another model's
  anomaly score, label-confidence columns, `disease_category` (a target proxy),
  and acquisition metadata (scanner, staining, magnification) that could let the
  model learn the *site* instead of the biology.
- **Preprocessing** — `ColumnTransformer` (standardize numerics + one-hot encode
  categoricals) inside a `Pipeline`, fit on training folds only (no leakage). The
  original code scaled the full dataset before splitting.
- **Modelling** — Logistic Regression, Random Forest, Gradient Boosting compared
  with 5-fold cross-validated ROC AUC; the top model tuned via `RandomizedSearchCV`.
- **Evaluation** — accuracy, ROC AUC/curve, confusion matrix, feature importance,
  plus a multi-class cell-type model.

## Results

**Binary anomaly detection (tuned Random Forest):** ~0.97 accuracy, ~0.99 ROC AUC.
**Multi-class cell type (Random Forest):** ~0.95 accuracy.

| Model | CV ROC AUC |
|-------|-----------|
| Random Forest (tuned) | ~0.99 |
| Gradient Boosting | ~0.99 |
| Logistic Regression | ~0.85 |

Cell size, nucleus area, and granularity are the strongest anomaly predictors.

> Re-run the notebook to regenerate exact numbers and plots, then update this table.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Blood_cell_anomaly_detection.ipynb
```

## Key skills demonstrated

Leakage awareness (the most important takeaway), pipelines & `ColumnTransformer`,
cross-validated model selection, hyperparameter tuning, multi-class
classification, and result interpretation.

## Possible extensions

Class weights / SMOTE for rare cell types, SHAP explanations for individual
predictions, and grouped cross-validation by `dataset_source` to confirm the
model generalizes across acquisition sites.
