# Task 2: Machine Learning — Bank Customer Churn Prediction

Predict the probability of `Exited` (customer churn) for every customer in
`test.csv`, using a model trained on `train.csv`, and report the model's
predictive ability.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

- Python 3.10+ (developed on 3.11).
- **macOS only**: XGBoost needs the OpenMP runtime — `brew install libomp`.
- Runtime: ~1 minute. All results are reproducible (fixed `random_state=42`).

## Files

| File | What it is |
|---|---|
| `main.py` | One-click pipeline: load → EDA plots → train → evaluate → export CSVs |
| `churn_analysis.ipynb` | Exploration notebook with the full EDA narrative and reasoning |
| `requirements.txt` | Dependencies |
| `train.csv` / `test.csv` | Provided data (165,034 / 110,023 rows) |

### Outputs (created by `python main.py`)

| Output | Content |
|---|---|
| `predictions.csv` | All original `test.csv` columns + **`Exited_Prob`** (predicted churn probability) + **`Exited_Pred`** (binary prediction at threshold 0.35) |
| `metrics.csv` | ROC AUC, F1, Precision, Recall, Accuracy and the decision threshold |
| `confusion_matrix.csv` | Labelled 2×2 confusion matrix |
| `plots/*.png` | 7 figures: EDA (01–04) and model evaluation (05–07) |

> **Where do the evaluation numbers come from?** `test.csv` has no labels, so
> the model cannot be evaluated on it. All metrics are computed on a
> **held-out validation set**: a stratified 20% split of `train.csv`
> (33,007 rows) that the model never sees during training.

## Method

### 1. EDA findings (see notebook / `plots/01–04`)

- **Class imbalance** — 21.2% churn. Accuracy alone is misleading; evaluation
  uses ROC AUC, F1, precision/recall instead.
- **Age** — churned customers are clearly older on average.
- **NumOfProducts is non-monotonic** — churn rate is 35% with 1 product,
  only **6%** with 2, but **88%** with 3–4. A single linear coefficient cannot
  express this U-shape.
- **Correlated explanatory variables** — e.g. `Geography_Germany` × `Balance`
  correlate at 0.54 (German customers almost always hold a positive balance).
- **Weak features** — `CreditScore`, `EstimatedSalary`, `Tenure`, `HasCrCard`
  distributions almost fully overlap between the two classes.

### 2. Model choice — driven by the data characteristics

Non-linear / non-monotonic patterns, correlated mixed-type features, no missing
values, and a mid-sized tabular dataset (165k rows) are exactly the setting
where gradient-boosted trees excel. **XGBoost** was selected:

- Trees capture the NumOfProducts U-shape and feature interactions
  automatically (no manual dummy/binning engineering needed).
- Insensitive to feature scaling and multicollinearity.
- Native categorical support (`enable_categorical=True`) — `Geography` and
  `Gender` are fed directly as `category` dtype.

Hyperparameters are kept moderate and fixed
(`n_estimators=300, learning_rate=0.1, max_depth=6`) — no heavy tuning, since
the validation AUC (0.889) already sits near this dataset's practical ceiling.

### 3. Validation design

1. **Stratified 80/20 split** of `train.csv` — stratification preserves the
   21.2% churn rate on both sides.
2. Train on 80%; **all reported metrics/curves come from the 20% validation set**.
3. The learning curve (`plots/05`) monitors train vs. validation logloss per
   boosting round to check for overfitting.

### 4. Decision threshold

The model outputs probabilities. To also provide a hard classification, the
threshold was swept over 0.05–0.95 and chosen to **maximise F1 on the
validation set** → **t = 0.35** (`plots/07`). The optimum sits below 0.5
because of class imbalance: lowering the threshold trades a little precision
for substantially better recall on the minority (churn) class.

### 5. Final model

Retrained on **100% of `train.csv`** with identical hyperparameters, then used
to predict `Exited_Prob` for `test.csv` (110,023 rows).

## Results (validation set, 33,007 rows)

| Metric | Value |
|---|---|
| ROC AUC | **0.8885** |
| F1 @ t=0.35 | **0.6662** |
| Precision | 0.6616 |
| Recall | 0.6708 |
| Accuracy | 0.8578 |

Confusion matrix @ t = 0.35:

| | Pred: Stayed | Pred: Exited |
|---|---|---|
| **True: Stayed** | 23,627 | 2,396 |
| **True: Exited** | 2,299 | 4,685 |

**Interpretation.** Against a 21.2% churn base rate, precision of 0.66 is a
~3× lift: two of every three customers flagged by the model are true churners,
while capturing 67% of all churners. The residual error is dominated by the
information limit of the available features (no behavioural signals such as
complaints or transaction activity), not by model capacity.

## Plots

| File | Shows |
|---|---|
| `01_target_distribution.png` | Class imbalance (21.2% churn) |
| `02_numeric_distributions.png` | Numeric features by churn status (Age separates best) |
| `03_churn_rate_by_category.png` | Churn rate per category (NumOfProducts U-shape) |
| `04_correlation_heatmap.png` | Feature correlations incl. target |
| `05_learning_curve.png` | Train vs. validation logloss per boosting round |
| `06_roc_pr_curves.png` | ROC curve and Precision-Recall curve (validation) |
| `07_threshold_sweep.png` | F1 vs. decision threshold; optimum at 0.35 |

## Possible improvements

- Behavioural features (transaction frequency, complaints, product usage)
  would raise the ceiling more than any modelling change.
- Probability calibration (isotonic / Platt) if probabilities feed a
  cost-based decision.
- Cost-sensitive threshold: replace F1 with expected-cost minimisation once
  retention/churn costs are known.
- Light hyperparameter search (`max_depth`, `min_child_weight`, `subsample`)
  for marginal gains.
