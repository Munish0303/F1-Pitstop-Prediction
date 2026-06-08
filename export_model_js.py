"""
export_model_js.py
------------------
Export the trained XGBoost model to a compact JSON that runs in the browser, so
the app needs no Python backend (works on Streamlit Cloud).

Uses XGBoost's native save_model JSON (exact float32 thresholds) and re-implements
the tree-walk here, asserting it matches the real model's predict_proba to <1e-5
before writing — so the JS inference is provably correct.

Output: static/model.json
  { intercept, threshold, features:[names in index order], trees:[{sc,si,l,r}] }
"""
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
model     = joblib.load(os.path.join(BASE, "models", "xgb_pitstop.pkl"))
threshold = json.load(open(os.path.join(BASE, "models", "threshold.json")))["threshold"]
booster   = model.get_booster()
feat_order = list(booster.feature_names)

# ── Dump native model JSON (exact thresholds) ────────────────────────────────
tmp = os.path.join(tempfile.gettempdir(), "xgb_native.json")
booster.save_model(tmp)
native = json.load(open(tmp))
raw_trees = native["learner"]["gradient_booster"]["model"]["trees"]

trees = []
for t in raw_trees:
    trees.append({
        "sc": [float(x) for x in t["split_conditions"]],  # threshold OR leaf value
        "si": [int(x)   for x in t["split_indices"]],
        "l":  [int(x)   for x in t["left_children"]],      # -1 => leaf
        "r":  [int(x)   for x in t["right_children"]],
    })

f32 = np.float32

def walk(tree, vec):
    """x32 < cond -> left, else right. Leaf when left == -1 (cond = leaf value)."""
    i = 0
    while tree["l"][i] != -1:
        x = f32(vec[tree["si"][i]])
        i = tree["l"][i] if x < f32(tree["sc"][i]) else tree["r"][i]
    return tree["sc"][i]

def predict_js(vec, intercept):
    margin = intercept + sum(walk(t, vec) for t in trees)
    return 1.0 / (1.0 + np.exp(-margin))

# ── Verify against the real model ────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "test_fe.csv"))
X = df[feat_order].head(400).reset_index(drop=True)
vecs = X.to_numpy()

margin_true = booster.predict(xgb.DMatrix(X, feature_names=feat_order), output_margin=True)
leaf_sums = np.array([sum(walk(t, vecs[i]) for t in trees) for i in range(len(X))])
intercepts = margin_true - leaf_sums
intercept = float(np.median(intercepts))
print(f"intercept: median={intercept:.6f}  std={intercepts.std():.2e}")

proba_true = model.predict_proba(X)[:, 1]
proba_js   = np.array([predict_js(vecs[i], intercept) for i in range(len(X))])
max_err = float(np.max(np.abs(proba_true - proba_js)))
print(f"max |proba_true - proba_js| over {len(X)} rows = {max_err:.2e}")
assert max_err < 1e-5, "JS inference does NOT match the model!"
print("OK — browser inference matches the trained model.")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE, "static"), exist_ok=True)
out = {"intercept": intercept, "threshold": threshold,
       "features": feat_order, "trees": trees}
path = os.path.join(BASE, "static", "model.json")
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB, {len(trees)} trees)")
