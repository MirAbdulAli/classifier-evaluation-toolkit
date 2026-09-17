"""
model_eval_toolkit
===================
A small, reusable toolkit for evaluating binary classifiers properly:

- Stratified k-fold cross-validation (mean +/- std, not a single split)
- Confusion matrix visualization
- ROC curve + AUC
- Precision-Recall curve + Average Precision
- Side-by-side multi-model comparison helpers

Everything here is deliberately framework-thin: it wraps scikit-learn's
own building blocks (StratifiedKFold, cross_val_score, confusion_matrix,
roc_curve/roc_auc_score, precision_recall_curve) rather than re-implementing
them, because those implementations are the correct, tested reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# 1. Cross-validation
# ---------------------------------------------------------------------------
def run_cross_validation(model, X, y, k=5, scoring="accuracy", random_state=42):
    """
    Run stratified k-fold CV for a single model/metric and return a dict with
    the fold scores plus mean and std. Stratified folds matter most on
    imbalanced data: plain KFold can accidentally starve a fold of the
    minority class.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        "scoring": scoring,
        "k": k,
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
    }


def cross_validate_multi_metric(model, X, y, k=5, metrics=("accuracy", "f1"), random_state=42):
    """Run CV for several metrics at once, return {metric: result_dict}."""
    return {m: run_cross_validation(model, X, y, k=k, scoring=m, random_state=random_state) for m in metrics}


def print_cv_summary(name, cv_results):
    any_res = next(iter(cv_results.values()))
    print(f"\n[{name}] {any_res['k']}-fold CV results")
    for metric, res in cv_results.items():
        print(f"  {metric:10s}: {res['mean']:.4f} +/- {res['std']:.4f}  (folds: {np.round(res['scores'], 3)})")


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, ax=None, title="Confusion Matrix", labels=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    if ax is not None:
        ax.set_title(title)
    return cm


# ---------------------------------------------------------------------------
# 3. ROC curve
# ---------------------------------------------------------------------------
def get_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, thresholds, auc


def plot_roc_curves(models_scores, y_true, ax=None, title="ROC Curve"):
    """
    models_scores: dict {model_name: y_score (probability of positive class)}
    Plots all models on one axis for direct comparison.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    for name, y_score in models_scores.items():
        fpr, tpr, _, auc = get_roc(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


# ---------------------------------------------------------------------------
# 4. Precision-Recall curve
# ---------------------------------------------------------------------------
def get_pr(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return precision, recall, thresholds, ap


def plot_pr_curves(models_scores, y_true, ax=None, title="Precision-Recall Curve"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    baseline = np.mean(y_true)  # positive-class prevalence = no-skill precision
    for name, y_score in models_scores.items():
        precision, recall, _, ap = get_pr(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", linewidth=2)
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"No-skill baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax


# ---------------------------------------------------------------------------
# 5. Full report for one model
# ---------------------------------------------------------------------------
def full_report(name, y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"\n=== {name}: held-out test set ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Avg Prec : {ap:.4f}")
    print(classification_report(y_true, y_pred, digits=3))
    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "avg_precision": ap}
