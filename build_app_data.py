"""
build_app_data.py
-----------------
Bakes everything the browser app needs (besides the model + track shapes) into
one static file: per-circuit metadata + real race laps, the decision threshold,
and precomputed feature insights (importance + averaged directional effect +
descriptions + linkage).

Run when the model/data changes (needs the ML stack):
    python build_app_data.py   ->   static/app_data.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
model     = joblib.load(os.path.join(BASE, "models", "xgb_pitstop.pkl"))
threshold = json.load(open(os.path.join(BASE, "models", "threshold.json")))["threshold"]
features  = json.load(open(os.path.join(BASE, "models", "feature_names.json")))["features"]
aggs      = json.load(open(os.path.join(BASE, "models", "circuit_aggregates.json")))
TEST_FE   = os.path.join(BASE, "data", "test_fe.csv")

CIRCUIT_OVERTAKING = {
    'Bahrain Grand Prix': 2, 'Saudi Arabian Grand Prix': 2, 'Australian Grand Prix': 1,
    'Japanese Grand Prix': 1, 'Chinese Grand Prix': 2, 'Miami Grand Prix': 1,
    'Emilia Romagna Grand Prix': 1, 'Monaco Grand Prix': 0, 'Spanish Grand Prix': 1,
    'Canadian Grand Prix': 1, 'Austrian Grand Prix': 2, 'British Grand Prix': 1,
    'Hungarian Grand Prix': 0, 'Belgian Grand Prix': 2, 'Dutch Grand Prix': 0,
    'Italian Grand Prix': 2, 'Azerbaijan Grand Prix': 1, 'Singapore Grand Prix': 0,
    'United States Grand Prix': 1, 'Mexico City Grand Prix': 1, 'São Paulo Grand Prix': 2,
    'Las Vegas Grand Prix': 2, 'Qatar Grand Prix': 1, 'Abu Dhabi Grand Prix': 1,
    'French Grand Prix': 1,
}
CIRCUIT_IS_STREET = {
    'Monaco Grand Prix': 1, 'Azerbaijan Grand Prix': 1, 'Singapore Grand Prix': 1,
    'Miami Grand Prix': 1, 'Las Vegas Grand Prix': 1, 'Saudi Arabian Grand Prix': 1,
}
CIRCUIT_DEG_TIER = {
    'Belgian Grand Prix': 2, 'Bahrain Grand Prix': 2, 'Japanese Grand Prix': 2,
    'British Grand Prix': 2, 'Spanish Grand Prix': 2, 'Qatar Grand Prix': 1,
    'Las Vegas Grand Prix': 1, 'French Grand Prix': 1, 'Austrian Grand Prix': 1,
    'United States Grand Prix': 1, 'Abu Dhabi Grand Prix': 1, 'Italian Grand Prix': 1,
    'Chinese Grand Prix': 1, 'Hungarian Grand Prix': 1, 'Dutch Grand Prix': 1,
    'Miami Grand Prix': 0, 'Canadian Grand Prix': 0, 'Singapore Grand Prix': 0,
    'São Paulo Grand Prix': 0, 'Saudi Arabian Grand Prix': 0, 'Azerbaijan Grand Prix': 0,
    'Emilia Romagna Grand Prix': 0, 'Australian Grand Prix': 0, 'Mexico City Grand Prix': 0,
    'Monaco Grand Prix': 0,
}
RACE_LAPS = {
    'Bahrain Grand Prix': 57, 'Saudi Arabian Grand Prix': 50, 'Australian Grand Prix': 58,
    'Azerbaijan Grand Prix': 51, 'Miami Grand Prix': 57, 'Emilia Romagna Grand Prix': 63,
    'Monaco Grand Prix': 78, 'Spanish Grand Prix': 66, 'Canadian Grand Prix': 70,
    'Austrian Grand Prix': 71, 'British Grand Prix': 52, 'Hungarian Grand Prix': 70,
    'Belgian Grand Prix': 44, 'Dutch Grand Prix': 72, 'Italian Grand Prix': 53,
    'Singapore Grand Prix': 62, 'Japanese Grand Prix': 53, 'Qatar Grand Prix': 57,
    'United States Grand Prix': 56, 'Mexico City Grand Prix': 71, 'São Paulo Grand Prix': 71,
    'Las Vegas Grand Prix': 50, 'Abu Dhabi Grand Prix': 58, 'Chinese Grand Prix': 56,
    'French Grand Prix': 53,
}
GLOBAL_RACE_LAPS = 58

FEATURE_META = {
    "LapNumber":                ("Race state", "Lap number", "Current lap of the race.", []),
    "RaceProgress":             ("Race state", "Race progress", "How far through the race we are (lap ÷ total laps, 0–1).", ["LapNumber", "race_laps"]),
    "Stint":                    ("Race state", "Stint number", "Which set of tyres this is (1st stint, 2nd, …). Later stints mean a pit has already happened.", []),
    "Position":                 ("Position",   "Track position", "Current running position (P1 = leading).", []),
    "position_vs_start":        ("Position",   "Positions vs start", "Places gained (+) or lost (−) since the start. Losing places can push a team toward an alternative strategy.", []),
    "TyreLife":                 ("Tyres",      "Tyre age", "Laps completed on the current set of tyres.", []),
    "Compound_enc":             ("Tyres",      "Compound", "Tyre compound fitted (soft/medium/hard/intermediate/wet). Softer tyres wear out sooner.", []),
    "cumulative_degradation_clean": ("Tyres", "Cumulative degradation", "Total lap-time lost this stint to tyre wear (seconds).", []),
    "LapTime_Delta_Clean":      ("Pace",       "Lap-time delta", "Change in lap time vs the driver's recent clean pace (seconds). Rising = tyres fading.", []),
    "laptime_pct_above_median": ("Pace",       "% above median pace", "How far this lap is above the field's median lap time (%).", []),
    "laptime_trend":            ("Pace",       "Lap-time trend", "Short-term trend in lap times. A strong upward trend is the clearest 'tyres are done' signal — the model's top feature.", []),
    "deg_trend":                ("Tyres",      "Degradation trend", "Whether tyre degradation is accelerating.", []),
    "car_tier_enc":             ("Car",        "Car tier", "Car competitiveness (top / midfield / back). Affects strategic stakes.", []),
    "is_wet_lap":               ("Weather",    "Wet lap", "Whether the lap is run in wet conditions.", []),
    "rainfall_any":             ("Weather",    "Rainfall", "Any rain falling during the lap.", []),
    "track_temp_mean":          ("Weather",    "Track temperature", "Track surface temperature (°C). Hotter tracks degrade tyres faster.", []),
    "humidity_mean":            ("Weather",    "Humidity", "Air humidity (%).", []),
    "wind_speed_mean":          ("Weather",    "Wind speed", "Wind speed (m/s) — affects car balance and tyre stress.", []),
    "neutralised_lap":          ("Race state", "Safety car / VSC", "Whether a safety car or virtual safety car is active — a cheap moment to pit.", []),
    "neutralised_prev":         ("Race state", "Prev lap neutralised", "Whether the previous lap was under safety car.", []),
    "circuit_overtaking":       ("Circuit",    "Overtaking ease", "How easy it is to pass here (0 very hard – 2 easy). Hard-to-pass tracks discourage pitting into traffic.", []),
    "circuit_is_street":        ("Circuit",    "Street circuit", "Whether this is a street circuit.", []),
    "circuit_deg_tier":         ("Circuit",    "Degradation tier", "How tough the circuit is on tyres (0 low – 2 high).", []),
    "circuit_avg_tyre_life":    ("Circuit",    "Avg tyre life", "Typical stint length seen historically at this circuit (laps).", []),
    "circuit_pit_rate":         ("Circuit",    "Pit rate", "Historical share of laps that are pit-in laps at this circuit.", []),
    "tyre_life_ratio":          ("Engineered", "Tyre-life ratio", "Tyre age ÷ the compound's typical stint length. Above 1.0 means the tyre is past its normal life.", ["TyreLife", "Compound_enc"]),
    "laps_past_cliff":          ("Engineered", "Laps past cliff", "How many laps the tyre is beyond its expected cliff (0 if still fresh enough).", ["TyreLife", "Compound_enc"]),
    "pit_window_distance":      ("Engineered", "Pit-window distance", "Distance from the mid-race pit window (|race progress − 0.5|).", ["RaceProgress"]),
    "in_pit_window":            ("Engineered", "In pit window", "Whether we're inside the prime 30–70% race-distance pit window.", ["RaceProgress"]),
    "deg_per_lap":              ("Engineered", "Degradation per lap", "Average tyre time-loss per lap this stint.", ["cumulative_degradation_clean", "TyreLife"]),
    "position_pressure":        ("Engineered", "Position pressure", "1 ÷ position — higher for the leaders, who have more at stake.", ["Position"]),
    "pos_loss_x_hard_circuit":  ("Engineered", "Lost places × hard track", "Places lost amplified by how hard the circuit is to overtake.", ["position_vs_start", "circuit_overtaking"]),
    "track_pos_premium":        ("Engineered", "Track-position premium", "Value of holding position at a hard-to-pass circuit (reluctance to pit).", ["circuit_overtaking", "Position"]),
    "is_soft_medium":           ("Engineered", "Is soft/medium", "Whether the compound is a soft or medium (faster-wearing).", ["Compound_enc"]),
    "tyre_stress":              ("Engineered", "Tyre stress", "Tyre age weighted by how soft the compound is.", ["TyreLife", "Compound_enc"]),
    "pit_opportunity":          ("Engineered", "Pit opportunity", "A discounted-stop opportunity under a safety car / VSC.", ["neutralised_lap"]),
    "is_stint_start":           ("Engineered", "Is stint start", "Whether the tyres are very fresh (≤ 2 laps) — won't pit again immediately.", ["TyreLife"]),
    "circuit_deg_x_tyre":       ("Engineered", "Circuit deg × tyre", "High-degradation circuit combined with old tyres = urgent stop.", ["circuit_deg_tier", "TyreLife"]),
}


def build_circuits():
    table = {}
    for name, avg_tyre in aggs["avg_tyre"].items():
        table[name] = {
            "name": name,
            "circuit_overtaking":    CIRCUIT_OVERTAKING.get(name, 1),
            "circuit_is_street":     CIRCUIT_IS_STREET.get(name, 0),
            "circuit_deg_tier":      CIRCUIT_DEG_TIER.get(name, 1),
            "circuit_avg_tyre_life": avg_tyre,
            "circuit_pit_rate":      aggs["pit_rate"].get(name, aggs["global_pit_rate"]),
            "race_laps":             RACE_LAPS.get(name, GLOBAL_RACE_LAPS),
        }
    return table


def feature_insights():
    df = pd.read_csv(TEST_FE)
    stats = {}
    for f in features:
        col = df[f] if f in df.columns else pd.Series([0.0])
        p10, p50, p90 = (float(x) for x in np.percentile(col, [10, 50, 90]))
        lo, hi = float(col.min()), float(col.max())
        if p10 == p90:
            p10, p90 = lo, hi
        stats[f] = (p10, p50, p90)

    sample = df.sample(min(500, len(df)), random_state=0)[features].reset_index(drop=True)
    base_prob = float(model.predict_proba(sample)[:, 1].mean())
    imp = dict(zip(features, (float(x) for x in model.feature_importances_)))
    imp_sum = sum(imp.values()) or 1.0

    out = []
    for f in features:
        low = sample.copy();  low[f]  = stats[f][0]
        high = sample.copy(); high[f] = stats[f][2]
        diff = model.predict_proba(high)[:, 1] - model.predict_proba(low)[:, 1]
        group, label, desc, parents = FEATURE_META.get(f, ("Other", f, "", []))
        out.append({
            "name": f, "group": group, "label": label, "description": desc,
            "derived_from": parents,
            "importance": round(imp[f] / max(imp.values()), 4),
            "importance_pct": round(100 * imp[f] / imp_sum, 2),
            "effect": round(float(diff.mean()), 4),
            "effect_abs": round(float(np.abs(diff).mean()), 4),
        })
    return base_prob, out


if __name__ == "__main__":
    base_prob, insights = feature_insights()
    data = {
        "circuits": build_circuits(),
        "threshold": threshold,
        "base_prob": round(base_prob, 4),
        "features": insights,
    }
    os.makedirs(os.path.join(BASE, "static"), exist_ok=True)
    path = os.path.join(BASE, "static", "app_data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB, "
          f"{len(data['circuits'])} circuits, {len(insights)} features)")
