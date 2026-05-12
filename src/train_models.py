"""Train Logistic Regression and Decision Tree on the OULAD modeling table.

Outputs (reports/ and models/):
  - classification reports (text)
  - confusion matrix plots
  - ROC curve (binary target)
  - cross-validation summary
  - grid-searched best models (.pkl)

Run from project root:
    python src/train_models.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CATEGORICAL = [
    "gender", "region", "highest_education", "imd_band",
    "age_band", "disability",
]
DROP_COLS = [
    "id_student", "code_module", "code_presentation",
    "final_result", "target_multiclass", "target_binary",
    # Leaky: did_unregister IS the withdrawn flag; days_registered encodes the same
    "did_unregister", "days_registered",
]
TARGET_COL = "target_binary"          # 1 = Withdrawn (at risk), 0 = all others
TARGET_LABELS = ["Not at risk", "At risk (Withdrawn)"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_parquet(PROCESSED / "modeling_table.parquet")

    # Encode categoricals
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    y = df[TARGET_COL].values
    X = df.drop(columns=DROP_COLS).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, MODELS / "scaler.pkl")

    feature_names = df.drop(columns=DROP_COLS).columns.tolist()
    return X_scaled, X, y, feature_names, scaler


def stratified_split(X, y, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def print_report(name, y_test, y_pred, y_prob=None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=TARGET_LABELS))
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")

    # Save report text
    report_text = classification_report(y_test, y_pred, target_names=TARGET_LABELS)
    (REPORTS / f"{name.lower().replace(' ', '_')}_report.txt").write_text(report_text)


def save_confusion_matrix(name, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=TARGET_LABELS,
        colorbar=False, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fig.savefig(REPORTS / f"{name.lower().replace(' ', '_')}_cm.png", dpi=150)
    plt.close(fig)


def save_roc_curves(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_test, y_prob in results:
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC Curves — Binary Risk Prediction")
    plt.tight_layout()
    fig.savefig(REPORTS / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"\nROC curve saved → {REPORTS}/roc_curves.png")


def cross_validate(name, model, X, y, cv=5):
    cv_scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1",
    )
    print(f"\n  {name} — {cv}-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return cv_scores


# ── Logistic Regression ────────────────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test, X_all, y_all):
    print("\nGrid-searching Logistic Regression …")
    param_grid = {
        "C":        [0.01, 0.1, 1, 10],
        "penalty":  ["l1", "l2"],
        "solver":   ["liblinear"],
        "max_iter": [500],
    }
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    print_report("Logistic Regression", y_test, y_pred, y_prob)
    save_confusion_matrix("Logistic Regression", y_test, y_pred)
    cross_validate("Logistic Regression", best, X_all, y_all)

    joblib.dump(best, MODELS / "logistic_regression.pkl")
    print(f"  Model saved → {MODELS}/logistic_regression.pkl")

    return best, y_test, y_prob


# ── Decision Tree ──────────────────────────────────────────────────────────────

def train_decision_tree(X_train, X_test, y_train, y_test, X_all, y_all, feature_names):
    print("\nGrid-searching Decision Tree …")
    param_grid = {
        "max_depth":        [4, 6, 8, 12, None],
        "min_samples_leaf": [10, 20, 50],
        "criterion":        ["gini", "entropy"],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    print_report("Decision Tree", y_test, y_pred, y_prob)
    save_confusion_matrix("Decision Tree", y_test, y_pred)
    cross_validate("Decision Tree", best, X_all, y_all)

    # Feature importance plot (top 15)
    importances = pd.Series(best.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    top15.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Decision Tree — Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(REPORTS / "dt_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot saved → {REPORTS}/dt_feature_importance.png")

    joblib.dump(best, MODELS / "decision_tree.pkl")
    print(f"  Model saved → {MODELS}/decision_tree.pkl")

    return best, y_test, y_prob


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading and preparing data …")
    X_scaled, X_raw, y, feature_names, scaler = load_and_prepare()

    X_train, X_test, y_train, y_test = stratified_split(X_scaled, y)
    print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"Class balance (test) — at-risk: {y_test.mean():.2%}")

    lr_model, lr_y_test, lr_y_prob = train_logistic_regression(
        X_train, X_test, y_train, y_test, X_scaled, y
    )
    dt_model, dt_y_test, dt_y_prob = train_decision_tree(
        X_train, X_test, y_train, y_test, X_scaled, y, feature_names
    )

    save_roc_curves([
        ("Logistic Regression", lr_y_test, lr_y_prob),
        ("Decision Tree",       dt_y_test, dt_y_prob),
    ])

    joblib.dump(scaler, MODELS / "scaler.pkl")
    print(f"Scaler saved → {MODELS}/scaler.pkl")
    print("\nAll done. Artifacts in reports/ and models/")


if __name__ == "__main__":
    main()
