"""
demo.py
=======
Applies eval_toolkit to a deliberately imbalanced binary classification
problem (roughly 95% / 5%, similar to fraud/anomaly detection), comparing
Logistic Regression vs. Random Forest with:
  - stratified 5-fold CV (accuracy + F1, mean +/- std)
  - confusion matrices on a held-out test set
  - ROC curves + AUC (both models, same axes)
  - Precision-Recall curves + Average Precision (both models, same axes)
  - a demonstration of accuracy being misleading on imbalanced data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from eval_toolkit import (
    cross_validate_multi_metric,
    print_cv_summary,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    full_report,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Build an imbalanced dataset (~95% negative / 5% positive)
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=6000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_clusters_per_class=2,
    weights=[0.95, 0.05],   # class imbalance
    flip_y=0.01,
    class_sep=0.9,
    random_state=RANDOM_STATE,
)

print(f"Dataset shape: {X.shape}")
print(f"Class balance: {np.bincount(y)} -> "
      f"{100 * np.bincount(y) / len(y)} % (neg, pos)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 2. Define two models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE),
}

# A trivial "always predict majority class" baseline, to make the
# accuracy-is-misleading point concrete.
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"\n--- Naive baseline ---\n"
      f"'Always predict majority class' test accuracy: {dummy_acc:.4f} "
      f"(this model NEVER finds a single positive case)")

# ---------------------------------------------------------------------------
# 3. Stratified k-fold cross-validation (k=5), accuracy + F1
# ---------------------------------------------------------------------------
cv_results = {}
for name, model in models.items():
    cv_results[name] = cross_validate_multi_metric(
        model, X_train, y_train, k=5, metrics=("accuracy", "f1"), random_state=RANDOM_STATE
    )
    print_cv_summary(name, cv_results[name])

# ---------------------------------------------------------------------------
# 4. Fit on full training set, predict on held-out test set
# ---------------------------------------------------------------------------
test_preds = {}
test_scores = {}
test_reports = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    test_preds[name] = y_pred
    test_scores[name] = y_score
    test_reports[name] = full_report(name, y_test, y_pred, y_score)

# ---------------------------------------------------------------------------
# 5. Confusion matrices (side by side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, y_pred) in zip(axes, test_preds.items()):
    plot_confusion_matrix(y_test, y_pred, ax=ax, title=name, labels=["Negative", "Positive"])
fig.suptitle("Confusion Matrices — Held-out Test Set", fontsize=13)
fig.tight_layout()
fig.savefig("confusion_matrices.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. ROC curves, both models on same axes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
plot_roc_curves(test_scores, y_test, ax=ax, title="ROC Curve — Model Comparison")
fig.tight_layout()
fig.savefig("roc_curve.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Precision-Recall curves, both models on same axes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
plot_pr_curves(test_scores, y_test, ax=ax, title="Precision-Recall Curve — Model Comparison")
fig.tight_layout()
fig.savefig("pr_curve.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Accuracy-is-misleading demonstration
# ---------------------------------------------------------------------------
print("\n=== Why accuracy alone is misleading here ===")
print(f"Naive 'always negative' baseline accuracy : {dummy_acc:.4f}")
for name in models:
    print(f"{name} test accuracy                    : {test_reports[name]['accuracy']:.4f}  "
          f"| F1: {test_reports[name]['f1']:.4f}  | ROC-AUC: {test_reports[name]['roc_auc']:.4f}")

print("\nDone. Saved: confusion_matrices.png, roc_curve.png, pr_curve.png")
