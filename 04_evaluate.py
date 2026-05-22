"""
04_evaluate.py
--------------
Evaluates saved model on held-out test set.
Prints full metrics and saves 4 plots to outputs/.
"""
# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────


import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
)

TEST_FE_PATH = f"{BASE_DIR}/data/test_fe.csv"
MODEL_PATH   = f"{BASE_DIR}/models/xgb_pitstop.pkl"
THRESH_PATH  = f"{BASE_DIR}/models/threshold.json"
FEAT_PATH    = f"{BASE_DIR}/models/feature_names.json"
OUTPUT_DIR   = f"{BASE_DIR}/outputs"

TARGET  = "PitNextLap"
EXCLUDE = ["Year"]


def load_data(path, feature_names):
    df   = pd.read_csv(path)
    drop = [c for c in EXCLUDE + [TARGET] if c in df.columns]
    X    = df.drop(columns=drop)[feature_names]
    y    = df[TARGET]
    return X, y


def plot_confusion_matrix(y_true, y_pred, path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Pit Next", "Pit Next"],
                yticklabels=["No Pit Next", "Pit Next"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — PitNextLap")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e10600", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — PitNextLap Classifier")
    ax.legend(); plt.tight_layout(); fig.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_pr(y_true, y_prob, threshold, path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#15151e", lw=2, label=f"AP = {ap:.4f}")
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(recall[idx], precision[idx], s=100, color="#e10600", zorder=5,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(); plt.tight_layout(); fig.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_feature_importance(model, feature_names, path, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[idx][::-1], color="#e10600")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Importance (gain)"); ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(THRESH_PATH) as f: threshold = json.load(f)["threshold"]
    with open(FEAT_PATH)   as f: feature_names = json.load(f)["features"]

    X_test, y_test = load_data(TEST_FE_PATH, feature_names)
    model          = joblib.load(MODEL_PATH)

    print(f"Test shape : {X_test.shape}  |  Threshold: {threshold:.4f}\n")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["No Pit Next", "Pit Next"]))

    roc_auc = roc_auc_score(y_test, y_prob)
    ap      = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"Avg Precision (PR-AUC): {ap:.4f}")
    print(f"F1 (positive class)  : {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc(y_test, y_prob,              os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_pr(y_test, y_prob, threshold,    os.path.join(OUTPUT_DIR, "pr_curve.png"))
    plot_feature_importance(model, list(X_test.columns),
                                          os.path.join(OUTPUT_DIR, "feature_importance.png"))


if __name__ == "__main__":
    evaluate()
