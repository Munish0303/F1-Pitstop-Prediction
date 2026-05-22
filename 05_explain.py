"""
05_explain.py
-------------
SHAP-based explainability on real test set rows.
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
import shap

TEST_FE_PATH = f"{BASE_DIR}/data/test_fe.csv"
MODEL_PATH   = f"{BASE_DIR}/models/xgb_pitstop.pkl"
FEAT_PATH    = f"{BASE_DIR}/models/feature_names.json"
OUTPUT_DIR   = f"{BASE_DIR}/outputs"

TARGET       = "PitNextLap"
EXCLUDE      = ["Year"]
SHAP_MAX     = 3000


def load_data(path, feature_names):
    df   = pd.read_csv(path)
    drop = [c for c in EXCLUDE + [TARGET] if c in df.columns]
    X    = df.drop(columns=drop)[feature_names]
    y    = df[TARGET]
    return X, y, df


def explain():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FEAT_PATH) as f: feature_names = json.load(f)["features"]

    X_test, y_test, df_full = load_data(TEST_FE_PATH, feature_names)
    model = joblib.load(MODEL_PATH)

    # Stratified sample: all pit-next laps + random stay laps
    pit_rows  = df_full[df_full[TARGET] == 1]
    stay_rows = df_full[df_full[TARGET] == 0]
    n_stay    = min(SHAP_MAX - len(pit_rows), len(stay_rows))
    sampled   = pd.concat([pit_rows, stay_rows.sample(n_stay, random_state=0)],
                          ignore_index=True)
    X_sample  = sampled.drop(columns=[TARGET] + EXCLUDE, errors="ignore")[feature_names]

    print(f"SHAP sample: {len(X_sample)} rows  "
          f"({len(pit_rows)} pit-next laps + {n_stay} stay laps)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Bar plot ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shap_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Beeswarm — PitNextLap Probability")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")

    # ── Waterfall: highest confidence pit-next prediction ─────────────────────
    y_prob  = model.predict_proba(X_test)[:, 1]
    pit_idx = int(np.argmax(y_prob))
    X_single = X_test.iloc[[pit_idx]]
    sv       = explainer.shap_values(X_single)

    exp = shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=X_single.iloc[0].values,
        feature_names=list(X_single.columns),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False, max_display=15)
    plt.title(
        f"Waterfall — Most Confident PitNextLap Prediction  "
        f"(Lap {int(X_test.iloc[pit_idx]['LapNumber'])}, "
        f"TyreLife={int(X_test.iloc[pit_idx]['TyreLife'])}, "
        f"prob={y_prob[pit_idx]:.3f})"
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shap_waterfall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    explain()
