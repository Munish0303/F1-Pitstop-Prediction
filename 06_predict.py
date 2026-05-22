"""
06_predict.py
-------------
Inference script. Loads real rows from test set — no hardcoded values.

Usage:
    python 06_predict.py              # show predictions on real test laps
    python 06_predict.py --csv f.csv  # batch predict on any CSV
    python 06_predict.py --n 10       # show 10 examples instead of 5
"""
# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────


import argparse
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH   = f"{BASE_DIR}/models/xgb_pitstop.pkl"
THRESH_PATH  = f"{BASE_DIR}/models/threshold.json"
FEAT_PATH    = f"{BASE_DIR}/models/feature_names.json"
TEST_FE_PATH = f"{BASE_DIR}/data/test_fe.csv"

TARGET  = "PitNextLap"
EXCLUDE = ["Year"]

COMPOUND_LABEL = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTERMEDIATE", 4: "WET"}
CAR_TIER_LABEL = {3: "top", 2: "mid", 1: "back", 0: "unknown"}


def load_test(feature_names):
    df   = pd.read_csv(TEST_FE_PATH)
    drop = [c for c in EXCLUDE if c in df.columns]
    return df.drop(columns=drop)


def predict_row(row, model, threshold, feature_names):
    X    = pd.DataFrame([{k: row.get(k, 0) for k in feature_names}])
    prob = model.predict_proba(X)[0, 1]
    return {
        "probability": round(float(prob), 4),
        "decision":    "PIT NEXT LAP 🔴" if prob >= threshold else "STAY OUT 🟢",
    }


def print_pred(label, row, result):
    compound = COMPOUND_LABEL.get(int(row.get("Compound_enc", 1)), "?")
    tier     = CAR_TIER_LABEL.get(int(row.get("car_tier_enc", 1)), "?")
    actual   = "PIT NEXT" if row.get(TARGET, -1) == 1 else "STAY"
    print(f"\n  {label}")
    print(f"    Lap {int(row['LapNumber'])}  "
          f"|  TyreLife={int(row['TyreLife'])}  |  Compound={compound}  "
          f"|  Position={int(row['Position'])}  |  CarTier={tier}")
    print(f"    Actual={actual}  →  Prob={result['probability']}  →  {result['decision']}")


def predict_csv(csv_path, model, threshold, feature_names):
    df      = pd.read_csv(csv_path)
    missing = [c for c in feature_names if c not in df.columns]
    for c in missing: df[c] = 0
    if missing: print(f"Warning: missing columns filled with 0: {missing}")
    probs          = model.predict_proba(df[feature_names])[:, 1]
    df["pit_prob"] = probs.round(4)
    df["decision"] = (probs >= threshold).map({True: "PIT_NEXT", False: "STAY"})
    out = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"Predictions saved → {out}")
    print(df[["pit_prob", "decision"]].head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--n",   type=int, default=5)
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    with open(THRESH_PATH) as f: threshold     = json.load(f)["threshold"]
    with open(FEAT_PATH)   as f: feature_names = json.load(f)["features"]

    if args.csv:
        predict_csv(args.csv, model, threshold, feature_names)
    else:
        df = load_test(feature_names)

        print(f"\nTarget       : PitNextLap (will driver pit on NEXT lap?)")
        print(f"Threshold    : {threshold:.4f}")
        print(f"Test rows    : {len(df):,}  |  Actual pit-next laps: {(df[TARGET]==1).sum()}")
        print("=" * 65)

        # Actual pit-next laps
        pit_rows = df[df[TARGET] == 1].head(args.n)
        print(f"\n{'─'*65}\n  ACTUAL PIT-NEXT LAPS (n={len(pit_rows)})\n{'─'*65}")
        for i, (_, row) in enumerate(pit_rows.iterrows(), 1):
            print_pred(f"Pit-Next Example {i}", row, predict_row(row, model, threshold, feature_names))

        # Actual stay laps
        stay_rows = df[df[TARGET] == 0].sample(args.n, random_state=None)
        print(f"\n{'─'*65}\n  ACTUAL STAY LAPS (n={len(stay_rows)})\n{'─'*65}")
        for i, (_, row) in enumerate(stay_rows.iterrows(), 1):
            print_pred(f"Stay Example {i}", row, predict_row(row, model, threshold, feature_names))

        # Top 5 highest confidence predictions
        X_all  = df.drop(columns=[TARGET], errors="ignore")
        probs  = model.predict_proba(X_all[feature_names])[:, 1]
        df     = df.copy(); df["prob"] = probs
        top5   = df.nlargest(5, "prob")
        print(f"\n{'─'*65}\n  TOP 5 HIGHEST CONFIDENCE PIT-NEXT PREDICTIONS\n{'─'*65}")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            result = {"probability": round(row["prob"], 4),
                      "decision": "PIT NEXT LAP 🔴" if row["prob"] >= threshold else "STAY OUT 🟢"}
            print_pred(f"Top {i}", row, result)

        print()
