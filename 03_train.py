"""
03_train.py
-----------
Trains XGBoost classifier on PitNextLap target.
Uses SMOTE for class imbalance and tunes threshold on F1.
"""
# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────


import pandas as pd
import numpy as np
import os
import joblib
import json

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score
)

TRAIN_FE_PATH = f"{BASE_DIR}/data/train_fe.csv"
MODEL_PATH    = f"{BASE_DIR}/models/xgb_pitstop.pkl"
THRESH_PATH   = f"{BASE_DIR}/models/threshold.json"
FEATURES_PATH = f"{BASE_DIR}/models/feature_names.json"

TARGET  = "PitNextLap"
EXCLUDE = ["Year"]  # kept for split grouping, dropped before training


def load_data(path: str):
    df   = pd.read_csv(path)
    drop = [c for c in EXCLUDE + [TARGET] if c in df.columns]
    X    = df.drop(columns=drop)
    y    = df[TARGET]
    return X, y


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx  = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train():
    X, y = load_data(TRAIN_FE_PATH)
    print(f"Train shape  : {X.shape}")
    print(f"Target rate  : {y.mean():.3%}  ({y.sum():,} pit-next-lap laps)")
    print(f"Features used: {list(X.columns)}\n")

    # ── SMOTE oversampling ────────────────────────────────────────────────────
    # Fill intentional NaNs with sentinel -999 before SMOTE
    # (LapTime_Delta_Clean and cumulative_degradation_clean are NaN on slow laps)
    # XGBoost still learns the pattern; -999 acts as explicit "slow lap" signal
    X_smote = X.fillna(-999)
    sm        = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_smote, y)
    print(f"After SMOTE: {X_res.shape}  |  Target rate: {y_res.mean():.3%}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,       # SMOTE already balances
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    # ── Cross-validation on original imbalanced train set ────────────────────
    print("\nCross-validating (5-fold stratified) on original train set ...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # ── Final fit on SMOTE-balanced data ──────────────────────────────────────
    model.fit(X_res, y_res, verbose=False)

    # ── Threshold tuning on original train set ────────────────────────────────
    y_prob_train          = model.predict_proba(X)[:, 1]
    best_thresh, best_f1  = find_best_threshold(y, y_prob_train)
    print(f"\nBest threshold : {best_thresh:.4f}  (train F1={best_f1:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": best_thresh}, f)

    with open(FEATURES_PATH, "w") as f:
        json.dump({"features": list(X.columns)}, f)

    print(f"\nModel saved    → {MODEL_PATH}")
    print(f"Threshold saved→ {THRESH_PATH}")
    print(f"Features saved → {FEATURES_PATH}")
    return model, best_thresh


if __name__ == "__main__":
    train()
