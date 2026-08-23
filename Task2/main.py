"""
Task 2: Machine Learning — Bank Customer Churn Prediction
==========================================================
Predicts the probability of `Exited` for every customer in test.csv,
using an XGBoost classifier trained on train.csv.

Run:
    python main.py

Inputs  (same folder): train.csv, test.csv
Outputs (same folder):
    predictions.csv       test.csv + Exited_Prob + Exited_Pred
    metrics.csv           evaluation metrics (on the held-out validation set)
    confusion_matrix.csv  2x2 confusion matrix (on the held-out validation set)
    plots/*.png           EDA and evaluation plots

Full exploratory analysis and reasoning: see churn_analysis.ipynb / README.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: figures are saved to files, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------- config ----
ID_COLS = ["id", "CustomerId", "Surname"]      # identifiers, not features
TARGET = "Exited"
CAT_COLS = ["Geography", "Gender"]
SEED = 42

XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    tree_method="hist", enable_categorical=True,
    random_state=SEED, n_jobs=-1, eval_metric="logloss",
)

PLOT_DIR = Path("plots")


# ------------------------------------------------------------- EDA plots ----
def make_eda_plots(train: pd.DataFrame) -> None:
    # 1. target distribution (class imbalance: 21.2% churn)
    counts = train[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["Stayed (0)", "Exited (1)"], counts.values, color=["C0", "C1"])
    for i, v in enumerate(counts.values):
        ax.annotate(f"{v:,} ({v / len(train):.1%})", (i, v), ha="center", va="bottom")
    ax.set_title("Target distribution")
    ax.set_ylabel("Customers")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. numeric feature distributions by churn status
    num_cols = ["Age", "CreditScore", "Balance", "EstimatedSalary", "Tenure"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, col in zip(axes.flat, num_cols):
        bins = np.arange(-0.5, 11.5, 1) if col == "Tenure" else 40
        for val, label in [(0, "Stayed"), (1, "Exited")]:
            ax.hist(train.loc[train[TARGET] == val, col], bins=bins,
                    density=True, alpha=0.55, label=label)
        ax.set_title(col)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric feature distributions by churn status")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_numeric_distributions.png", dpi=150)
    plt.close(fig)

    # 3. churn rate by categorical feature (note NumOfProducts: non-monotonic)
    cat_cols = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
    overall = train[TARGET].mean()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for ax, col in zip(axes, cat_cols):
        rate = train.groupby(col)[TARGET].mean()
        ax.bar(rate.index.astype(str), rate.values)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1)
        for i, v in enumerate(rate.values):
            ax.annotate(f"{v:.0%}", (i, v), ha="center", va="bottom", fontsize=9)
        ax.set_title(col)
        ax.margins(y=0.15)
    axes[0].set_ylabel("Churn rate")
    fig.suptitle(f"Churn rate by category (dashed = overall {overall:.1%})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_churn_rate_by_category.png", dpi=150)
    plt.close(fig)

    # 4. feature correlation (Geography one-hot expanded; Gender binary-encoded)
    enc = train.drop(columns=ID_COLS).copy()
    enc["Gender"] = (enc["Gender"] == "Male").astype(int)
    enc = pd.get_dummies(enc, columns=["Geography"], dtype=int)
    corr = enc.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (include target Exited)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ evaluation plots ----
def plot_learning_curve(model: XGBClassifier) -> None:
    hist = model.evals_result()  # validation_0 = training set, validation_1 = validation set
    rounds = range(1, len(hist["validation_0"]["logloss"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, hist["validation_0"]["logloss"], label="Train")
    ax.plot(rounds, hist["validation_1"]["logloss"], label="Validation")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Logloss")
    ax.set_title("Learning curve (logloss)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_learning_curve.png", dpi=150)
    plt.close(fig)


def plot_prediction_curves(y_val: pd.Series, proba_val: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    fpr, tpr, _ = roc_curve(y_val, proba_val)
    axes[0].plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_val, proba_val):.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (validation)")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_val, proba_val)
    axes[1].plot(rec, prec,
                 label=f"XGBoost (AP = {average_precision_score(y_val, proba_val):.3f})")
    axes[1].axhline(y_val.mean(), color="gray", linestyle="--", linewidth=1,
                    label=f"Baseline ({y_val.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (validation)")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_roc_pr_curves.png", dpi=150)
    plt.close(fig)


def sweep_threshold(y_val: pd.Series, proba_val: np.ndarray) -> float:
    """Pick the decision threshold that maximises F1 on the validation set."""
    thresholds = np.round(np.linspace(0.05, 0.95, 181), 3)
    f1s = np.array([f1_score(y_val, (proba_val >= t).astype(int)) for t in thresholds])
    best_t = float(thresholds[f1s.argmax()])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(thresholds, f1s)
    ax.axvline(best_t, color="C1", linestyle="--")
    ax.annotate(f"best t = {best_t:.2f}\nF1 = {f1s.max():.3f}",
                (best_t, f1s.max()), xytext=(best_t + 0.08, f1s.max() - 0.08))
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1 (validation)")
    ax.set_title("Threshold sweep")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_threshold_sweep.png", dpi=150)
    plt.close(fig)
    return best_t


# -------------------------------------------------------------- pipeline ----
def main() -> None:
    PLOT_DIR.mkdir(exist_ok=True)

    # 0. load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"train: {train.shape}   test: {test.shape}")

    # 1. EDA plots
    make_eda_plots(train)
    print(f"EDA plots saved to {PLOT_DIR}/")

    # 2. features / target (identifiers dropped; categoricals fed natively to XGBoost)
    X = train.drop(columns=ID_COLS + [TARGET]).copy()
    X[CAT_COLS] = X[CAT_COLS].astype("category")
    y = train[TARGET]

    # 3. stratified 80/20 split — all evaluation happens on the 20% validation set
    #    (test.csv has no labels, so it cannot be used for evaluation)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"training set:   {len(X_tr):,} rows (churn rate {y_tr.mean():.4f})")
    print(f"validation set: {len(X_val):,} rows (churn rate {y_val.mean():.4f})")

    # 4. train on the 80% split, monitor logloss on both sets for the learning curve
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    proba_val = model.predict_proba(X_val)[:, 1]

    # 5. evaluation on the validation set
    plot_learning_curve(model)
    plot_prediction_curves(y_val, proba_val)
    best_t = sweep_threshold(y_val, proba_val)
    y_pred = (proba_val >= best_t).astype(int)

    print(f"Validation ROC AUC : {roc_auc_score(y_val, proba_val):.4f}")
    print(f"Best threshold (max F1 on validation): {best_t:.2f}")
    print(f"Validation F1@{best_t:.2f} : {f1_score(y_val, y_pred):.4f}")

    # metrics.csv
    metrics = pd.DataFrame([
        ["Evaluated_on", f"validation set (20% of train, {len(y_val):,} rows)"],
        ["ROC_AUC",            round(roc_auc_score(y_val, proba_val), 4)],
        ["Decision_threshold", round(best_t, 2)],
        ["F1",                 round(f1_score(y_val, y_pred), 4)],
        ["Precision",          round(precision_score(y_val, y_pred), 4)],
        ["Recall",             round(recall_score(y_val, y_pred), 4)],
        ["Accuracy",           round(accuracy_score(y_val, y_pred), 4)],
    ], columns=["metric", "value"])
    metrics.to_csv("metrics.csv", index=False)

    # confusion_matrix.csv (2x2, labelled)
    cm_table = pd.DataFrame(confusion_matrix(y_val, y_pred),
                            index=["True: Stayed", "True: Exited"],
                            columns=["Pred: Stayed", "Pred: Exited"])
    cm_table.to_csv("confusion_matrix.csv")
    print("saved metrics.csv, confusion_matrix.csv")
    print(cm_table)

    # 6. predict test.csv with the SAME model that was evaluated above —
    #    the reported validation metrics describe exactly this model
    X_test = test.drop(columns=ID_COLS).copy()
    X_test[CAT_COLS] = X_test[CAT_COLS].astype("category")
    proba_test = model.predict_proba(X_test)[:, 1]

    output = test.copy()
    output["Exited_Prob"] = proba_test.round(4)
    output["Exited_Pred"] = (proba_test >= best_t).astype(int)
    output.to_csv("predictions.csv", index=False)
    print(f"saved predictions.csv ({len(output):,} rows); "
          f"predicted churners at t={best_t:.2f}: {output['Exited_Pred'].sum():,} "
          f"({output['Exited_Pred'].mean():.1%})")


if __name__ == "__main__":
    main()
