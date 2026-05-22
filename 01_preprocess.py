"""
01_preprocess.py
----------------
Loads raw F1 lap data, filters non-strategic laps, drops columns,
encodes categoricals, adds circuit type features, saves train/test splits.

TARGET : PitNextLap
SPLIT  : GroupShuffleSplit on Year
"""
# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import json

RAW_PATH   = f"{BASE_DIR}/data/f1_strategy_v4.csv"
TRAIN_PATH = f"{BASE_DIR}/data/train.csv"
TEST_PATH  = f"{BASE_DIR}/data/test.csv"

TARGET = "PitNextLap"

# ══════════════════════════════════════════════════════════════════════════════
# COLUMNS TO DROP
# ══════════════════════════════════════════════════════════════════════════════

LEAKY = [
    "PitStop",          # IS the current-lap pit event
    "pit_duration",     # post-event only; 99.9% null
    "compound_openf1",  # post-stint confirmed; 83.4% null
]

DEPRECATED = [
    "LapTime_Delta",            # CORRUPTED: post-pit shows -30s delta
    "Cumulative_Degradation",   # DEPRECATED: distorted baseline
    "Normalized_TyreLife",      # DEPRECATED: unclear normalisation
]

BROKEN = [
    "sc_lap",   # BUG: all zeros
    "vsc_lap",  # BUG: all zeros
]

IDENTIFIERS = [
    "Driver",       # string ID only
    "constructor",  # replaced by car_tier_enc
    "session_key",  # internal join key only
]
# Race kept temporarily for circuit feature extraction, dropped after

REDUNDANT = [
    "total_race_laps",      # constant per race; already in RaceProgress
    "race_median_laptime",  # normalisation base; not a feature
    "track_temp_max",       # collinear with track_temp_mean
    "air_temp_mean",        # r≈0.8 with track_temp_mean
    "tyre_age_at_start",    # 95.5% zeros; captured by TyreLife
    "true_tyre_age",        # = TyreLife + tyre_age_at_start; redundant
    "compound_hardness",    # identical ranking to Compound_enc; redundant
    "laps_to_end",          # redundant with RaceProgress
    "Position_Change",      # lap-to-lap noise; position_vs_start is better
    "start_position",       # static; fully captured by Position + position_vs_start
    "inferred_sc_vsc",      # 99.96% identical to neutralised_lap; keep neutralised_lap
    "red_flag_lap",         # used only as a ROW FILTER below; not a feature
    "LapTime (s)",          # raw laptime not cross-race comparable; laptime_pct is better
    "is_slow_lap",          # crude binary the model over-relied on; laptime_pct captures
                            # the same degradation signal more granularly without dominance
]

DROP_COLS = LEAKY + DEPRECATED + BROKEN + IDENTIFIERS + REDUNDANT

# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT ENCODINGS
# ══════════════════════════════════════════════════════════════════════════════

# Overtaking difficulty: 0=very hard, 1=moderate, 2=easy
CIRCUIT_OVERTAKING = {
    'Bahrain Grand Prix':           2,
    'Saudi Arabian Grand Prix':     2,
    'Australian Grand Prix':        1,
    'Japanese Grand Prix':          1,
    'Chinese Grand Prix':           2,
    'Miami Grand Prix':             1,
    'Emilia Romagna Grand Prix':    1,
    'Monaco Grand Prix':            0,
    'Spanish Grand Prix':           1,
    'Canadian Grand Prix':          1,
    'Austrian Grand Prix':          2,
    'British Grand Prix':           1,
    'Hungarian Grand Prix':         0,
    'Belgian Grand Prix':           2,
    'Dutch Grand Prix':             0,
    'Italian Grand Prix':           2,
    'Azerbaijan Grand Prix':        1,
    'Singapore Grand Prix':         0,
    'United States Grand Prix':     1,
    'Mexico City Grand Prix':       1,
    'São Paulo Grand Prix':         2,
    'Las Vegas Grand Prix':         2,
    'Qatar Grand Prix':             1,
    'Abu Dhabi Grand Prix':         1,
    'French Grand Prix':            1,
}

# Street circuit flag
CIRCUIT_IS_STREET = {
    'Monaco Grand Prix':            1,
    'Azerbaijan Grand Prix':        1,
    'Singapore Grand Prix':         1,
    'Miami Grand Prix':             1,
    'Las Vegas Grand Prix':         1,
    'Saudi Arabian Grand Prix':     1,
}

# Tyre degradation tier — data-driven from avg TyreLife per circuit
# avg_tyre_life < 12.5 = high(2), 12.5–16 = medium(1), >16 = low(0)
CIRCUIT_DEG_TIER = {
    'Belgian Grand Prix':           2,   # avg TyreLife 10.6
    'Bahrain Grand Prix':           2,   # avg TyreLife 10.8
    'Japanese Grand Prix':          2,   # avg TyreLife 12.1
    'British Grand Prix':           2,   # avg TyreLife 12.5
    'Spanish Grand Prix':           2,   # avg TyreLife 12.5
    'Qatar Grand Prix':             1,
    'Las Vegas Grand Prix':         1,
    'French Grand Prix':            1,
    'Austrian Grand Prix':          1,
    'United States Grand Prix':     1,
    'Abu Dhabi Grand Prix':         1,
    'Italian Grand Prix':           1,
    'Chinese Grand Prix':           1,
    'Hungarian Grand Prix':         1,
    'Dutch Grand Prix':             1,
    'Miami Grand Prix':             0,   # avg TyreLife 15.9
    'Canadian Grand Prix':          0,
    'Singapore Grand Prix':         0,
    'São Paulo Grand Prix':         0,
    'Saudi Arabian Grand Prix':     0,
    'Azerbaijan Grand Prix':        0,
    'Emilia Romagna Grand Prix':    0,
    'Australian Grand Prix':        0,
    'Mexico City Grand Prix':       0,
    'Monaco Grand Prix':            0,   # avg TyreLife 25.1
}


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add only the hand-mapped circuit features (fixed lookups — no leakage).
    The data-driven aggregates (circuit_avg_tyre_life, circuit_pit_rate) are
    computed AFTER the split, fitted on train only, in fit_circuit_aggregates().
    Race is kept here and dropped after aggregates are computed.
    """
    df["circuit_overtaking"] = df["Race"].map(CIRCUIT_OVERTAKING).fillna(1).astype(int)
    df["circuit_is_street"]  = df["Race"].map(CIRCUIT_IS_STREET).fillna(0).astype(int)
    df["circuit_deg_tier"]   = df["Race"].map(CIRCUIT_DEG_TIER).fillna(1).astype(int)
    print(f"  Added 3 hand-mapped circuit features (Race kept for aggregates)")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal lag/trend features within each driver's race.
    These use ONLY past laps (shift forward), so no future leakage.
    Requires Driver, Race, Year, LapNumber to still be present.

    Targets the model's main weakness: SC/VSC-driven pit stops and
    accelerating-degradation pit triggers, which static features miss.
    """
    df = df.sort_values(["Driver", "Race", "Year", "LapNumber"]).reset_index(drop=True)
    g = df.groupby(["Driver", "Race", "Year"])

    # Lap-on-lap pace change: a sudden slowdown often = in-lap before pit
    df["laptime_pct_prev"] = g["laptime_pct_above_median"].shift(1)
    df["laptime_trend"]    = (df["laptime_pct_above_median"] - df["laptime_pct_prev"]).fillna(0)

    # Degradation trend: is cumulative degradation accelerating?
    df["deg_prev"]   = g["cumulative_degradation_clean"].shift(1)
    df["deg_trend"]  = (df["cumulative_degradation_clean"] - df["deg_prev"]).fillna(0)

    # Was the previous lap neutralised? SC/VSC pits often happen 1 lap into the SC
    df["neutralised_prev"] = g["neutralised_lap"].shift(1).fillna(0).astype(int)

    # Clean up intermediate columns
    df = df.drop(columns=["laptime_pct_prev", "deg_prev"])
    print(f"  Added 3 lag/trend features: laptime_trend, deg_trend, neutralised_prev")
    return df


def fit_circuit_aggregates(train_df: pd.DataFrame):
    """
    Compute per-circuit aggregates from the TRAINING set only.
    Returns lookup dicts + global fallbacks (for circuits unseen in train).
    This prevents test-set target/feature info leaking into the features.
    """
    avg_tyre = train_df.groupby("Race")["TyreLife"].mean().round(2).to_dict()
    pit_rate = train_df.groupby("Race")[TARGET].mean().round(4).to_dict()
    # Global fallbacks for any circuit not present in the training split
    global_avg_tyre = round(train_df["TyreLife"].mean(), 2)
    global_pit_rate = round(train_df[TARGET].mean(), 4)
    return {
        "avg_tyre": avg_tyre,
        "pit_rate": pit_rate,
        "global_avg_tyre": global_avg_tyre,
        "global_pit_rate": global_pit_rate,
    }


def apply_circuit_aggregates(df: pd.DataFrame, agg: dict) -> pd.DataFrame:
    """Map the train-fitted aggregates onto a dataframe, then drop Race."""
    df["circuit_avg_tyre_life"] = (
        df["Race"].map(agg["avg_tyre"]).fillna(agg["global_avg_tyre"])
    )
    df["circuit_pit_rate"] = (
        df["Race"].map(agg["pit_rate"]).fillna(agg["global_pit_rate"])
    )
    df = df.drop(columns=["Race"])
    return df


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw: {df.shape}")

    # ── FILTER: remove red flag laps ──────────────────────────────────────────
    # Red flag = race stopped, tyre change is automatic/free, not a strategy decision
    # Lap times are meaningless on these laps
    before = len(df)
    df = df[df["red_flag_lap"] == 0].reset_index(drop=True)
    print(f"Filtered {before - len(df)} red flag laps → {len(df)} rows remaining")

    # ── Add hand-mapped circuit features (Race kept for train-only aggregates) ─
    df = add_circuit_features(df)

    # ── Add lag/trend features (needs Driver + LapNumber ordering) ────────────
    df = add_lag_features(df)

    # ── Drop columns ──────────────────────────────────────────────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"After dropping {len(cols_to_drop)} columns: {df.shape}")
    print(f"  Dropped: {cols_to_drop}")

    # ── Fill weather nulls (all 2022 rows) with median ────────────────────────
    for c in ["track_temp_mean", "humidity_mean", "rainfall_any", "wind_speed_mean"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # ── Fill Compound nulls (0.1%) with mode ──────────────────────────────────
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].fillna(df["Compound"].mode()[0])

    # ── LapTime_Delta_Clean and cumulative_degradation_clean: keep NaN ────────
    # Intentional — slow laps. XGBoost handles NaN natively.

    print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # ── Encode categoricals ───────────────────────────────────────────────────
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    if "Compound" in df.columns:
        df["Compound_enc"] = df["Compound"].map(compound_map).fillna(1).astype(int)
        df = df.drop(columns=["Compound"])

    tier_map = {"top": 3, "mid": 2, "back": 1, "unknown": 0}
    if "car_performance_tier" in df.columns:
        df["car_tier_enc"] = df["car_performance_tier"].map(tier_map).fillna(0).astype(int)
        df = df.drop(columns=["car_performance_tier"])

    print(f"\nFinal shape : {df.shape}")
    print(f"Target dist :\n{df[TARGET].value_counts()}")
    print(f"Target rate : {df[TARGET].mean():.3%}")
    return df


def split_and_save(df: pd.DataFrame):
    # Race is still present here — needed for fitting circuit aggregates
    X      = df.drop(columns=[TARGET])
    y      = df[TARGET]
    groups = df["Year"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # ── Fit circuit aggregates on TRAIN ONLY, then apply to both ──────────────
    # This prevents test-set TyreLife / target info leaking into the features.
    agg = fit_circuit_aggregates(train_df)
    train_df = apply_circuit_aggregates(train_df, agg)
    test_df  = apply_circuit_aggregates(test_df,  agg)
    print(f"\nFitted circuit aggregates on TRAIN only "
          f"({len(agg['avg_tyre'])} circuits), applied to both splits")

    os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
    # Persist aggregates so inference (06_predict.py) uses identical train-fitted values
    with open(f"{BASE_DIR}/models/circuit_aggregates.json", "w") as f:
        json.dump(agg, f, indent=2)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,   index=False)

    print(f"\nTrain : {train_df.shape} | Test : {test_df.shape}")
    print(f"Train {TARGET} : {train_df[TARGET].value_counts().to_dict()}")
    print(f"Test  {TARGET} : {test_df[TARGET].value_counts().to_dict()}")
    print(f"\nSaved → {TRAIN_PATH}")
    print(f"Saved → {TEST_PATH}")


if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    split_and_save(df)
