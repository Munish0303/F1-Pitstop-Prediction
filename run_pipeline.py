"""
run_pipeline.py
---------------
Runs the full pipeline end-to-end.
    python run_pipeline.py
"""

# ── SET THIS TO YOUR PROJECT ROOT ─────────────────────────────────────────────
BASE_DIR = "C:/Users/YourName/f1_pitstop_classifier"   # e.g. "C:/Users/you/f1_pitstop_classifier"
# ──────────────────────────────────────────────────────────────────────────────

import subprocess
import sys

STEPS = [
    ("Preprocessing",       f"{BASE_DIR}/01_preprocess.py"),
    ("Feature Engineering", f"{BASE_DIR}/02_feature_engineering.py"),
    ("Training",            f"{BASE_DIR}/03_train.py"),
    ("Evaluation",          f"{BASE_DIR}/04_evaluate.py"),
    ("SHAP Explainability", f"{BASE_DIR}/05_explain.py"),
]


def run(step_name, script):
    print(f"\n{'='*55}")
    print(f"  {step_name}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Step failed: {step_name}")
        sys.exit(1)
    print(f"✅ Done: {step_name}")


if __name__ == "__main__":
    for name, script in STEPS:
        run(name, script)
    print("\n🏁 Pipeline complete. Check outputs/ for plots.")
