# F1 Pit Stop Timing Classifier

A machine learning pipeline that predicts whether an F1 driver will pit on the **next lap**, built from lap-by-lap telemetry data across the 2022–2024 seasons. The project emphasises methodological rigour — leakage-free feature engineering, proper temporal train/test splitting, and transparent evaluation — over chasing headline accuracy numbers.

---

## What this predicts

**Target: `PitNextLap`** — will this driver pit on the next lap? (binary, 1 = yes)

This is a harder and more useful problem than predicting whether *this* lap is a pit lap. `PitNextLap=1` means the driver is on a normal racing lap, and the model has to infer from tyre degradation, race context, and circuit characteristics that a pit stop is imminent.

---

## Results

| Metric | Score | Notes |
|---|---|---|
| **ROC-AUC** | **0.9789** | Threshold-independent ranking quality |
| **PR-AUC** | **0.7870** | Primary metric for imbalanced data (baseline: 0.032) |
| **F1** | **0.7018** | At F1-optimal threshold (0.397) |
| Precision | 0.76 | When model says pit, correct 76% of the time |
| Recall | 0.65 | Catches 65% of actual pit laps |
| Accuracy | 0.98 | Misleading on its own — 97% of laps are non-pit |

**Why PR-AUC is the headline metric:** Only 3.2% of laps result in a pit stop. A model that always predicts "no pit" achieves 97.8% accuracy and 0 F1. PR-AUC measures model quality independently of class imbalance — a random classifier scores 0.032, this model scores 0.787.

### Baseline comparison

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Always predict "No Pit" | 0.500 | 0.032 | 0.000 |
| Logistic Regression | 0.901 | 0.312 | 0.411 |
| Random Forest (100 trees) | 0.961 | 0.631 | 0.612 |
| **XGBoost + SMOTE (this project)** | **0.979** | **0.787** | **0.702** |

### Performance by pit stop type

The model has two distinct operating modes depending on the *cause* of the pit stop:

| Pit type | Recall | Notes |
|---|---|---|
| Tyre-degradation driven | **74.4%** | Model understands tyre cliff well |
| Safety car / VSC driven | **27.2%** | Inherently harder — depends on accidents elsewhere on track |

SC/VSC pits are theoretically unpredictable from telemetry alone. The `laptime_trend` and `neutralised_prev` lag features were added specifically to improve this — at a lower threshold (0.15) SC-pit recall reaches 48.8%.

---

## Project structure

```
f1_pitstop_classifier/
├── data/
│   └── f1_strategy_v4.csv          # Raw data (not included — see Data section)
├── models/                          # Saved artifacts (auto-generated)
│   ├── xgb_pitstop.pkl
│   ├── threshold.json
│   ├── feature_names.json
│   └── circuit_aggregates.json      # Train-fitted circuit stats
├── outputs/                         # Plots (auto-generated)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── shap_bar.png
│   ├── shap_beeswarm.png
│   └── shap_waterfall.png
├── 01_preprocess.py                 # Filter, encode, lag features, train/test split
├── 02_feature_engineering.py        # Engineered features
├── 03_train.py                      # SMOTE + XGBoost + threshold tuning
├── 04_evaluate.py                   # Metrics + plots
├── 05_explain.py                    # SHAP explainability
├── 06_predict.py                    # Inference on real laps
├── run_pipeline.py                  # Run all steps end-to-end
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Place f1_strategy_v4.csv in data/
# Then run the full pipeline:
python run_pipeline.py

# Or predict on real laps from the test set:
python 06_predict.py

# Batch predict on a CSV:
python 06_predict.py --csv your_data.csv

# Show more examples:
python 06_predict.py --n 10
```

---

## Pipeline walkthrough

### `01_preprocess.py` — Cleaning and splitting

**Row filtering:** 66 red flag laps removed. Under a red flag the race is stopped and tyre changes are mandatory — there is no strategic decision being made, and lap times are meaningless.

**18 columns dropped**, grouped by reason:

| Reason | Columns |
|---|---|
| Leaky (post-event info) | `PitStop`, `pit_duration`, `compound_openf1` |
| Corrupted / deprecated | `LapTime_Delta` (−30s artefact), `Cumulative_Degradation`, `Normalized_TyreLife` |
| All-zero bug | `sc_lap`, `vsc_lap` (join failed in source data) |
| String identifiers | `Driver`, `constructor`, `session_key` |
| Redundant / collinear | `total_race_laps`, `race_median_laptime`, `track_temp_max`, `air_temp_mean`, `tyre_age_at_start`, `true_tyre_age`, `compound_hardness`, `laps_to_end`, `Position_Change`, `start_position`, `LapTime (s)`, `is_slow_lap` |

Notable drop: **`is_slow_lap`** — removed despite being the single most predictive feature (52% XGBoost gain). A model built around this feature learned to recognise in-lap characteristics after the decision was already made, not the strategy leading up to it. Removing it cut that dominance to 28% spread across meaningful degradation signals, with less than 0.3% loss in ROC-AUC.

**Lag/trend features** (built before `Driver` is dropped, ordered by `LapNumber`):

| Feature | What it captures |
|---|---|
| `laptime_trend` | Lap-on-lap pace change — sudden slowdown signals imminent pit (became #1 feature at 0.268 gain) |
| `deg_trend` | Is tyre degradation accelerating? Captures the cliff |
| `neutralised_prev` | Was the previous lap under SC/VSC? SC pits often happen one lap into the yellow |

**Circuit features** — encoded from `Race` string then dropped:

| Feature | Type | Description |
|---|---|---|
| `circuit_overtaking` | 0/1/2 | Hard (Monaco=0) to easy (Monza=2) |
| `circuit_deg_tier` | 0/1/2 | Low (Monaco=0) to high (Belgium=2) tyre wear |
| `circuit_is_street` | 0/1 | Street circuit flag |
| `circuit_avg_tyre_life` | float | Mean tyre life at this circuit — **fitted on train only** |
| `circuit_pit_rate` | float | Historical pit rate at this circuit — **fitted on train only** |

The last two are computed from the training set only and mapped onto the test set — preventing test-set information from leaking into the features.

**Train/test split:** `GroupShuffleSplit` on `Year`. Full seasons stay together in either split, preventing any race appearing in both train and test.

### `02_feature_engineering.py` — Engineered features

All 13 engineered features use only information available at prediction time:

| Feature | Formula / logic |
|---|---|
| `tyre_life_ratio` | `TyreLife / expected_compound_life` — data-driven cliff proximity (SOFT=15, MED=19, HARD=26 laps from observed median stint lengths) |
| `laps_past_cliff` | `max(0, TyreLife − expected_life)` — explicit overage laps |
| `deg_per_lap` | Degradation rate per lap |
| `tyre_stress` | `TyreLife × (2 − Compound_enc)` — older + softer = more stressed |
| `is_soft_medium` | Binary — soft/medium wear faster |
| `is_stint_start` | `TyreLife ≤ 2` — fresh rubber, unlikely to pit again |
| `pit_opportunity` | `= neutralised_lap` — SC/VSC = free pit window |
| `position_pressure` | `1 / Position` |
| `track_pos_premium` | `(2 − circuit_overtaking) × position_pressure` — hard overtaking + good position = reluctant to pit |
| `pos_loss_x_hard_circuit` | Position loss at hard-overtaking circuit → pressure for alternative strategy |
| `circuit_deg_x_tyre` | Circuit degradation level × tyre age |
| `pit_window_distance` | Distance from typical mid-race undercut window |
| `race_urgency` | `1 − RaceProgress` |

### `03_train.py` — Training

- **SMOTE oversampling:** 3.1% positive class balanced to 50/50 in training only — applied after the split to avoid cross-split contamination. NaN sentinel (`−999`) used for SMOTE compatibility; XGBoost receives NaN directly.
- **XGBoost:** `max_depth=8`, `n_estimators=500`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`. Depth-8 chosen by 5-fold CV comparison.
- **Threshold tuning:** Sweep across all thresholds via `precision_recall_curve`, select the one maximising F1 on the training set.

### `04_evaluate.py` — Evaluation

Saves 4 plots to `outputs/`: confusion matrix, ROC curve, precision-recall curve (with operating threshold marked), and XGBoost gain-based feature importance.

### `05_explain.py` — SHAP

`TreeExplainer` (exact Shapley values for trees) on a stratified 3,000-row sample — all 827 test pit laps plus 2,173 randomly sampled stay laps. Outputs: bar chart (global importance), beeswarm (per-feature direction and distribution), waterfall (single-prediction breakdown).

### `06_predict.py` — Inference

Loads real rows from `test_fe.csv`, shows predictions with full lap context, and supports `--csv path.csv` for batch prediction.

---

## Feature importance (top 10)

| Rank | Feature | Gain | Category |
|---|---|---|---|
| 1 | `laptime_trend` | 0.268 | Lag — pace change vs previous lap |
| 2 | `laptime_pct_above_median` | 0.111 | Tyre degradation signal |
| 3 | `is_stint_start` | 0.050 | Fresh tyre flag |
| 4 | `deg_trend` | 0.039 | Lag — degradation acceleration |
| 5 | `tyre_life_ratio` | 0.035 | Tyre cliff proximity |
| 6 | `circuit_pit_rate` | 0.032 | Circuit historical strategy |
| 7 | `cumulative_degradation_clean` | 0.031 | Total stint degradation |
| 8 | `TyreLife` | 0.024 | Raw tyre age |
| 9 | `Stint` | 0.024 | Race strategy context |
| 10 | `neutralised_lap` | 0.023 | Safety car / VSC flag |

---

## Methodology notes

**Temporal split — not random shuffle.** A random 80/20 split would let the model train on lap 30 of a race and test on lap 15 of the same race — leaking race-specific conditions across splits. `GroupShuffleSplit` on `Year` ensures entire seasons stay in one split.

**Circuit aggregates fitted on train only.** `circuit_avg_tyre_life` and `circuit_pit_rate` are computed on the training set and mapped onto the test set, not recomputed on the full dataset. This prevents per-circuit test-set target information from entering the features.

**Intentional NaN preservation.** `LapTime_Delta_Clean` and `cumulative_degradation_clean` are null on slow laps by design. They are not imputed — XGBoost learns a separate split direction for missing values, so the missingness itself becomes a signal.

**`is_slow_lap` deliberately dropped.** Investigation showed 35.8% of `PitNextLap=1` rows are already flagged as slow laps, meaning the feature was partly describing in-lap conditions rather than predicting them. Removing it forced the model to learn genuine degradation signals.

---

## Known limitations

**SC/VSC pit recall is 27% at the operating threshold.** These depend on random incidents elsewhere on track. Lowering the threshold to 0.15 improves this to ~49% at the cost of more false alarms.

**No gap-to-car-ahead data.** The undercut decision depends heavily on the gap to the car in front. This is the most important missing feature — teams regularly pit regardless of tyre age to undercut a rival.

**SMOTE creates synthetic samples.** On genuinely novel conditions (new compounds, regulation changes) performance may degrade relative to test metrics.

**2022 weather data is null.** OpenF1 weather coverage starts 2023. Weather features have no signal for 2022 races.

---

## Version history

| Version | Change | PR-AUC |
|---|---|---|
| v1–v4 | Initial exploration, wrong target (`PitStop`) | — |
| v5 | Switched to correct target `PitNextLap` | 0.712 |
| v6 | Added circuit type features | 0.738 |
| v7 | Red flag filter, dropped redundant columns | 0.765 |
| v8 | Hyperparameter tuning (depth 8, 500 trees) | 0.777 |
| v9 | Dropped `is_slow_lap` — healthier feature distribution | 0.769 |
| v10 | Leakage fix — circuit aggregates fitted on train only | 0.773 |
| **v11** | **Lag/trend features — `laptime_trend`, `deg_trend`, `neutralised_prev`** | **0.787** |

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## Data

`f1_strategy_v4.csv` contains 96,336 laps across the 2022–2024 F1 seasons sourced from the OpenF1 API, not included due to file size. Place it in `data/` before running.

**96,336 raw → 96,270 after red flag filter → 72,314 train / 23,956 test**
