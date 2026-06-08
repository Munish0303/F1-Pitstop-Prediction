"""Verify every UI control actually affects the model prediction, end-to-end,
through the running Flask API. Also checks RaceProgress is per-circuit."""
import json, urllib.request

API = "http://127.0.0.1:5000/api/predict"

def predict(**over):
    # Baseline chosen near the decision boundary so each input can show effect.
    body = dict(circuit="Bahrain Grand Prix", LapNumber=25, Stint=1, Position=8,
                position_vs_start=-2, Compound_enc=0, TyreLife=12,
                cumulative_degradation_clean=2.5, deg_trend=0.6,
                LapTime_Delta_Clean=1.0, laptime_pct_above_median=3.0,
                laptime_trend=1.0, car_tier_enc=2, track_temp_mean=35,
                humidity_mean=50, wind_speed_mean=2, is_wet_lap=0,
                rainfall_any=0, neutralised_lap=0, neutralised_prev=0)
    body.update(over)
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

# (key, [values across the UI slider range])
SWEEPS = [
    ("LapNumber",                    [1, 15, 30, 45, 57]),
    ("Stint",                        [1, 2, 3, 4, 5]),
    ("Position",                     [1, 5, 10, 15, 20]),
    ("position_vs_start",            [-12, -6, 0, 6, 12]),
    ("Compound_enc",                 [0, 1, 2, 3, 4]),
    ("TyreLife",                     [2, 12, 22, 32, 42]),
    ("cumulative_degradation_clean", [0, 1.5, 3, 4.5, 6]),
    ("deg_trend",                    [-1, 0, 0.8, 1.5, 2]),
    ("LapTime_Delta_Clean",          [-3, 0, 1.5, 3, 5]),
    ("laptime_pct_above_median",     [-2, 1, 4, 8, 12]),
    ("laptime_trend",                [-2, 0, 1, 2, 3]),
    ("car_tier_enc",                 [1, 2, 3]),
    ("track_temp_mean",              [10, 25, 40, 55]),
    ("humidity_mean",                [10, 40, 70, 100]),
    ("wind_speed_mean",              [0, 4, 8, 12]),
    ("is_wet_lap",                   [0, 1]),
    ("neutralised_lap",              [0, 1]),
]

print("baseline prob:", predict()["probability"])
print(f"\n{'control':<30} {'min%':>6} {'max%':>6} {'d%':>6}  effect")
print("-" * 62)
for key, vals in SWEEPS:
    probs = []
    for v in vals:
        kw = {key: v}
        if key == "is_wet_lap":        kw["rainfall_any"] = v
        if key == "neutralised_lap":   kw["neutralised_prev"] = v
        probs.append(predict(**kw)["probability"])
    lo, hi = min(probs), max(probs)
    delta = (hi - lo) * 100
    effect = "MOVES" if delta >= 0.5 else "flat (no effect)"
    print(f"{key:<30} {lo*100:6.1f} {hi*100:6.1f} {delta:6.1f}  {effect}")

# RaceProgress per-circuit check: same lap, different circuits -> different progress
print("\nRaceProgress is per-circuit (lap 39):")
for gp in ["Monaco Grand Prix", "Belgian Grand Prix", "Italian Grand Prix"]:
    r = predict(circuit=gp, LapNumber=39)
    print(f"  {gp:<22} lap 39/{r['race_laps']:<3} -> progress {r['race_progress']}")
