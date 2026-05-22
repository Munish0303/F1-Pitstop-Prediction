"""
02_feature_engineering.py
--------------------------
Builds engineered features from clean train/test splits.
All features use only information available at prediction time.

This version removes redundant features and adds high-value tyre-cliff
and pit-window features based on data analysis.
"""
# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────


import pandas as pd
import numpy as np
import os

TRAIN_PATH    = f"{BASE_DIR}/data/train.csv"
TEST_PATH     = f"{BASE_DIR}/data/test.csv"
TRAIN_FE_PATH = f"{BASE_DIR}/data/train_fe.csv"
TEST_FE_PATH  = f"{BASE_DIR}/data/test_fe.csv"

TARGET = "PitNextLap"

# Data-driven median stint length per compound (from actual dataset analysis)
# Compound_enc: SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4
EXPECTED_STINT_LIFE = {0: 15, 1: 19, 2: 26, 3: 14, 4: 7}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ══════════════════════════════════════════════════════════════════════════
    # TYRE CLIFF FEATURES (new — high value)
    # ══════════════════════════════════════════════════════════════════════════

    # How close is the tyre to its typical cliff for THIS compound?
    # >1.0 means the tyre is past its normal stint length → very likely to pit
    df["expected_life"]   = df["Compound_enc"].map(EXPECTED_STINT_LIFE).fillna(19)
    df["tyre_life_ratio"] = df["TyreLife"] / df["expected_life"]

    # Laps past the expected cliff (0 if still within normal range)
    df["laps_past_cliff"] = (df["TyreLife"] - df["expected_life"]).clip(lower=0)

    # ══════════════════════════════════════════════════════════════════════════
    # PIT WINDOW FEATURES (new)
    # ══════════════════════════════════════════════════════════════════════════

    # Distance from typical undercut pit window (RaceProgress 0.35–0.65)
    # 0 if inside window, positive if outside
    window_center = 0.5
    df["pit_window_distance"] = (df["RaceProgress"] - window_center).abs()

    # In the prime pit window?
    df["in_pit_window"] = ((df["RaceProgress"] >= 0.30) &
                           (df["RaceProgress"] <= 0.70)).astype(int)

    # ══════════════════════════════════════════════════════════════════════════
    # DEGRADATION FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # Degradation rate per lap
    df["deg_per_lap"] = df["cumulative_degradation_clean"] / df["TyreLife"].clip(lower=1)

    # ══════════════════════════════════════════════════════════════════════════
    # POSITION / STRATEGY FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # Position pressure (top positions = more strategic stakes)
    df["position_pressure"] = 1 / df["Position"].clip(lower=1)

    # Losing positions at a hard-to-overtake circuit = pressure for alt strategy
    df["pos_loss_x_hard_circuit"] = (
        df["position_vs_start"].clip(upper=0).abs() * (2 - df["circuit_overtaking"])
    )

    # Track position premium: hard overtaking + good position = reluctant to pit
    df["track_pos_premium"] = (2 - df["circuit_overtaking"]) * df["position_pressure"]

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOUND FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # Soft/Medium wear faster
    df["is_soft_medium"] = (df["Compound_enc"] <= 1).astype(int)

    # Tyre stress: older + softer = more stressed
    df["tyre_stress"] = df["TyreLife"] * (2 - df["Compound_enc"].clip(0, 2))

    # ══════════════════════════════════════════════════════════════════════════
    # SAFETY CAR / SLOW LAP FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # Pit opportunity under SC/VSC
    df["pit_opportunity"] = df["neutralised_lap"].astype(int)

    # Fresh rubber — unlikely to pit again immediately
    df["is_stint_start"] = (df["TyreLife"] <= 2).astype(int)

    # ══════════════════════════════════════════════════════════════════════════
    # CIRCUIT INTERACTION FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    # High-deg circuit + old tyres = urgent pit
    df["circuit_deg_x_tyre"] = df["circuit_deg_tier"] * df["TyreLife"]

    # Cleanup helper column
    df = df.drop(columns=["expected_life"])

    print(f"Features after engineering: {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    for in_path, out_path in [(TRAIN_PATH, TRAIN_FE_PATH), (TEST_PATH, TEST_FE_PATH)]:
        df = pd.read_csv(in_path)
        df = add_features(df)
        os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}  shape={df.shape}")
