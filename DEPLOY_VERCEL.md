# Deploying to Vercel

This app is **backend-free** — the trained XGBoost model is exported to JSON and
runs entirely in the browser (see `export_model_js.py`). So the whole thing ships
as a **static site**. The Streamlit app (`app.py`) is untouched and still works
with `streamlit run app.py`; the Vercel deploy is an additional, parallel path.

## What gets deployed

Everything in [`public/`](public/) — a self-contained static build:

```
public/
  index.html               tab shell (Predictor | Feature insights)
  predictor.html           the 3D predictor UI  (= templates/index.html + data)
  insights.html            the feature-insights UI (= templates/features.html + data)
  assets/model.js          window.MODEL   (the in-browser XGBoost model)
  assets/app_data.js       window.APPDATA (circuits + feature insights, filtered)
  assets/circuits_geo.js   window.GEO     (3D track outlines, filtered)
```

`vercel.json` tells Vercel to serve `public/` with no build step.

## Regenerating the build

`public/` is generated from the existing static assets + templates by a
stdlib-only script (no ML libraries needed):

```bash
python build_vercel.py
```

Run this whenever you retrain the model or change the templates. (First refresh
the static inputs with `export_model_js.py` / `build_app_data.py` /
`build_circuits_geo.py`, which *do* need the ML stack — see `requirements-dev.txt`.)

## Circuit filtering (recent calendar only)

Grands Prix no longer on the recent F1 calendar are hidden from the circuit
picker. This is controlled by `NON_RECENT_CIRCUITS` in `build_vercel.py`:

```python
NON_RECENT_CIRCUITS = {
    "French Grand Prix",   # last held 2022
}
```

Every other circuit in the dataset is on the 2024/2025 calendar. To hide more
(or restore one), edit that set and re-run `python build_vercel.py`.

## Deploy

### Option A — Vercel dashboard (Git import)
1. Push this repo to GitHub/GitLab/Bitbucket.
2. In Vercel: **Add New → Project → Import** this repo.
3. Framework Preset: **Other**. Leave Build Command empty and Output Directory
   as `public` (already set in `vercel.json`).
4. **Deploy.**

### Option B — Vercel CLI
```bash
npm i -g vercel
vercel        # preview deploy
vercel --prod # production deploy
```

## Local preview (matches what Vercel serves)
```bash
python -m http.server 8000 --directory public
# open http://localhost:8000
```

> Note: the predictor loads Three.js from a CDN (`unpkg.com`), so the page needs
> internet access at runtime — same as the original app.
