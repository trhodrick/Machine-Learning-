# School Measles (MMR) Vaccination Rates

A **regression** project on ~46,000 U.S. schools, predicting MMR vaccination
coverage and exploring where it falls below the herd-immunity threshold. The
standout skill here is **cleaning a genuinely messy real-world dataset**.

## Dataset

School-level vaccination data (WSJ / data.world), 14 columns including state,
school type, enrollment, academic year, MMR rate, overall rate, and exemption
percentages (religious / medical / personal).

Key data-quality issues handled:

- `-1` "not reported" sentinels in `mmr`/`overall` (converted to `NaN`).
- Heavy missingness — exemption columns are 70–86% missing, `type` ~59% missing.
- Academic-year strings like `"2017-18"` parsed to a numeric year.

> The CSV (`all-measles-rates.csv`) is not committed. Place it in the project root
> before running.

## Approach

- **Cleaning** — sentinel handling, year parsing, and documented column choices.
- **Leakage-aware feature selection** — drops `overall` (a near-duplicate of the
  target), the mostly-missing/mechanistic exemption columns, and identifier-like
  text fields; keeps `state`, `type`, `enroll`, `year_start`.
- **EDA** — MMR distribution vs the 95% threshold, coverage by state and school
  type, exemptions vs coverage.
- **Preprocessing** — `Pipeline` with imputation + scaling + one-hot encoding,
  fit on training folds only.
- **Modelling** — Linear Regression vs Random Forest vs Gradient Boosting,
  compared with 5-fold CV R²; top model tuned via `RandomizedSearchCV`.
- **Evaluation** — R², MAE, RMSE, predicted-vs-actual plot, feature importance.

## Results

Predicting coverage from school/geography alone is intentionally hard, so the
honest takeaway is methodological rather than a headline R². Tree ensembles beat
the linear baseline; **state** and **enrollment** are the most informative
features.

> Re-run the notebook to fill in exact R²/MAE/RMSE and update this section.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook measles_vaccination_rates.ipynb
```

## Key skills demonstrated

Real-world data cleaning (sentinels, missingness, string parsing), leakage-aware
feature selection, imputation pipelines, cross-validated model selection, and
honest interpretation of a hard problem.

## Possible extensions

Reframe as classification (below vs above 95%), join county-level socioeconomic
data, or model exemption rates directly on the reported subset.
