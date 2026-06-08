"""
app.py
------
Flask backend for the interactive F1 pit-stop predictor.

Serves a Three.js page (templates/index.html) and exposes:
  GET  /api/circuits   -> per-circuit metadata (overtaking, street, deg tier,
                          avg tyre life, pit rate)
  POST /api/predict    -> takes the user-controlled base inputs, recomputes the
                          engineered features EXACTLY like 02_feature_engineering.py,
                          runs the real trained XGBoost model, returns the
                          pit-next-lap probability + decision.

Run:
    python app.py
    open http://127.0.0.1:5000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "xgb_pitstop.pkl")
THRESH_PATH = os.path.join(BASE_DIR, "models", "threshold.json")
FEAT_PATH   = os.path.join(BASE_DIR, "models", "feature_names.json")
AGG_PATH    = os.path.join(BASE_DIR, "models", "circuit_aggregates.json")
TEST_FE_PATH = os.path.join(BASE_DIR, "data", "test_fe.csv")

# ── Load model + config once at startup ───────────────────────────────────────
model     = joblib.load(MODEL_PATH)
threshold = json.load(open(THRESH_PATH))["threshold"]
features  = json.load(open(FEAT_PATH))["features"]
aggs      = json.load(open(AGG_PATH))

# Median stint life per compound — copied verbatim from 02_feature_engineering.py
# Compound_enc: SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4
EXPECTED_STINT_LIFE = {0: 15, 1: 19, 2: 26, 3: 14, 4: 7}

# ── Hand-mapped circuit features — copied verbatim from 01_preprocess.py ───────
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


# Scheduled race distance (laps) per circuit — used to turn LapNumber into
# RaceProgress (lap / total laps), the way it was computed in training.
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


def build_circuit_table():
    """One dict per circuit with every circuit-level model feature."""
    table = {}
    for name, avg_tyre in aggs["avg_tyre"].items():
        table[name] = {
            "name": name,
            "circuit_overtaking":     CIRCUIT_OVERTAKING.get(name, 1),
            "circuit_is_street":      CIRCUIT_IS_STREET.get(name, 0),
            "circuit_deg_tier":       CIRCUIT_DEG_TIER.get(name, 1),
            "circuit_avg_tyre_life":  avg_tyre,
            "circuit_pit_rate":       aggs["pit_rate"].get(name, aggs["global_pit_rate"]),
            "race_laps":              RACE_LAPS.get(name, GLOBAL_RACE_LAPS),
        }
    return table


CIRCUITS = build_circuit_table()


def engineer(base: dict) -> dict:
    """Recompute the engineered features from base inputs.

    Mirrors 02_feature_engineering.py exactly so the model sees the same
    feature distribution it was trained on.
    """
    f = dict(base)

    tyre_life  = float(f["TyreLife"])
    compound   = int(f["Compound_enc"])
    progress   = float(f["RaceProgress"])
    position   = float(f["Position"])

    # Tyre cliff
    expected_life = EXPECTED_STINT_LIFE.get(compound, 19)
    f["tyre_life_ratio"] = tyre_life / expected_life
    f["laps_past_cliff"] = max(0.0, tyre_life - expected_life)

    # Pit window
    f["pit_window_distance"] = abs(progress - 0.5)
    f["in_pit_window"] = int(0.30 <= progress <= 0.70)

    # Degradation
    f["deg_per_lap"] = float(f["cumulative_degradation_clean"]) / max(1.0, tyre_life)

    # Position / strategy
    pos_clip = max(1.0, position)
    f["position_pressure"] = 1.0 / pos_clip
    pos_loss = abs(min(0.0, float(f["position_vs_start"])))
    f["pos_loss_x_hard_circuit"] = pos_loss * (2 - float(f["circuit_overtaking"]))
    f["track_pos_premium"] = (2 - float(f["circuit_overtaking"])) * f["position_pressure"]

    # Compound
    f["is_soft_medium"] = int(compound <= 1)
    f["tyre_stress"] = tyre_life * (2 - min(2, max(0, compound)))

    # Safety car / slow lap
    f["pit_opportunity"] = int(f["neutralised_lap"])
    f["is_stint_start"] = int(tyre_life <= 2)

    # Circuit interaction
    f["circuit_deg_x_tyre"] = float(f["circuit_deg_tier"]) * tyre_life

    return f


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

# group, human label, plain-English description, and which inputs each feature is
# built from (empty = a raw input the user sets directly).
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
    # ── engineered ──
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


def compute_feature_insights():
    """Data-driven importance + directional sensitivity for every model feature.

    - importance: the model's own gain-based feature importance.
    - effect: the AVERAGE change in pit probability, across many real laps, when
      the feature is moved from its typical-low (10th pct) to typical-high (90th
      pct) value while every other feature stays as it really was. Averaging over
      the distribution (not one fixed baseline) gives each feature's true typical
      influence. Signed mean = direction; mean |Δ| = magnitude shown in the bar.
    """
    df = pd.read_csv(TEST_FE_PATH)
    stats = {}
    for f in features:
        col = df[f] if f in df.columns else pd.Series([0.0])
        p10, p50, p90 = (float(x) for x in np.percentile(col, [10, 50, 90]))
        lo, hi = float(col.min()), float(col.max())
        if p10 == p90:                      # binary / low-variance → use full range
            p10, p90 = lo, hi
        stats[f] = (p10, p50, p90)

    # Representative sample of real laps to average the effect over.
    sample = df.sample(min(500, len(df)), random_state=0)[features].reset_index(drop=True)
    base_prob = float(model.predict_proba(sample)[:, 1].mean())

    imp = dict(zip(features, (float(x) for x in model.feature_importances_)))
    imp_sum = sum(imp.values()) or 1.0

    out = []
    for f in features:
        low = sample.copy();  low[f]  = stats[f][0]
        high = sample.copy(); high[f] = stats[f][2]
        diff = model.predict_proba(high)[:, 1] - model.predict_proba(low)[:, 1]
        signed  = float(diff.mean())
        abs_eff = float(np.abs(diff).mean())
        group, label, desc, parents = FEATURE_META.get(f, ("Other", f, "", []))
        out.append({
            "name": f, "group": group, "label": label, "description": desc,
            "derived_from": parents,
            "importance": round(imp[f] / max(imp.values()), 4),
            "importance_pct": round(100 * imp[f] / imp_sum, 2),
            "effect": round(signed, 4),
            "effect_abs": round(abs_eff, 4),
        })
    return base_prob, out


# Precompute once at startup (cheap: ~76 model calls)
BASE_PROB, FEATURE_INSIGHTS = compute_feature_insights()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/features")
def features_page():
    return render_template("features.html")


@app.route("/api/features")
def api_features():
    return jsonify({
        "features": FEATURE_INSIGHTS,
        "base_prob": round(BASE_PROB, 4),
        "threshold": round(threshold, 4),
    })


@app.route("/api/circuits")
def api_circuits():
    return jsonify({
        "circuits": CIRCUITS,
        "threshold": threshold,
        "global_avg_tyre": aggs["global_avg_tyre"],
        "global_pit_rate": aggs["global_pit_rate"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)

    circuit = CIRCUITS.get(payload.get("circuit"))
    if circuit is None:
        return jsonify({"error": "unknown circuit"}), 400

    # RaceProgress is derived from this circuit's real race distance, exactly as
    # in training (lap / total laps) — not a flat 70 for every track.
    lap_number  = float(payload.get("LapNumber", 1))
    race_laps   = circuit["race_laps"]
    race_prog   = min(1.0, max(0.0, lap_number / race_laps))

    # Merge user inputs with the selected circuit's fixed features
    base = {
        "LapNumber":                    lap_number,
        "RaceProgress":                 race_prog,
        "Stint":                        float(payload.get("Stint", 1)),
        "TyreLife":                     float(payload.get("TyreLife", 1)),
        "LapTime_Delta_Clean":          float(payload.get("LapTime_Delta_Clean", 0.0)),
        "laptime_pct_above_median":     float(payload.get("laptime_pct_above_median", 0.0)),
        "cumulative_degradation_clean": float(payload.get("cumulative_degradation_clean", 0.0)),
        "Position":                     float(payload.get("Position", 10)),
        "position_vs_start":            float(payload.get("position_vs_start", 0.0)),
        "is_wet_lap":                   int(payload.get("is_wet_lap", 0)),
        "neutralised_lap":              int(payload.get("neutralised_lap", 0)),
        "track_temp_mean":             float(payload.get("track_temp_mean", 30.0)),
        "humidity_mean":                float(payload.get("humidity_mean", 50.0)),
        "rainfall_any":                 int(payload.get("rainfall_any", 0)),
        "wind_speed_mean":              float(payload.get("wind_speed_mean", 2.0)),
        "laptime_trend":                float(payload.get("laptime_trend", 0.0)),
        "deg_trend":                    float(payload.get("deg_trend", 0.0)),
        "neutralised_prev":             int(payload.get("neutralised_prev", 0)),
        "Compound_enc":                 int(payload.get("Compound_enc", 1)),
        "car_tier_enc":                 int(payload.get("car_tier_enc", 2)),
        # circuit-level
        "circuit_overtaking":           circuit["circuit_overtaking"],
        "circuit_is_street":            circuit["circuit_is_street"],
        "circuit_deg_tier":             circuit["circuit_deg_tier"],
        "circuit_avg_tyre_life":        circuit["circuit_avg_tyre_life"],
        "circuit_pit_rate":             circuit["circuit_pit_rate"],
    }

    full = engineer(base)
    X = pd.DataFrame([{k: full.get(k, 0) for k in features}])[features]
    prob = float(model.predict_proba(X)[0, 1])

    return jsonify({
        "probability":  round(prob, 4),
        "threshold":    round(threshold, 4),
        "decision":     "PIT NEXT LAP" if prob >= threshold else "STAY OUT",
        "pit":          bool(prob >= threshold),
        "race_progress": round(race_prog, 3),
        "race_laps":     race_laps,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
