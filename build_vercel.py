"""
build_vercel.py
---------------
Generate a fully static, backend-free build of the F1 Pit-Stop Predictor for
Vercel (or any static host) into ./public.

The trained XGBoost model already runs in the browser (see export_model_js.py),
so there is nothing to run server-side. This script just bakes the existing
static assets + templates into a self-contained static site:

    public/
      index.html          tab shell (Predictor | Feature insights)
      predictor.html      templates/index.html  with data wired in
      insights.html       templates/features.html with data wired in
      assets/model.js     window.MODEL  = <static/model.json>
      assets/app_data.js  window.APPDATA = <static/app_data.json>  (filtered)
      assets/circuits_geo.js  window.GEO = <static/circuits_geo.json> (filtered)

Circuit filtering
-----------------
Grands Prix that are no longer on the recent F1 calendar are hidden from the
circuit picker (and their 3D track shapes are dropped). Edit NON_RECENT_CIRCUITS
to adjust. Currently only the French GP (last held 2022) is excluded; every
other circuit appears on the 2024/2025 calendars.

This script is stdlib-only — no ML libraries needed. Run it after regenerating
the static assets (export_model_js.py / build_app_data.py / build_circuits_geo.py):

    python build_vercel.py
"""
import json
import os
import shutil

BASE     = os.path.dirname(os.path.abspath(__file__))
STATIC   = os.path.join(BASE, "static")
TPL      = os.path.join(BASE, "templates")
OUT      = os.path.join(BASE, "public")
ASSETS   = os.path.join(OUT, "assets")

# Grands Prix dropped from the recent F1 calendar — hidden from the app.
# (Everything else in the dataset is on the 2024/2025 calendar.)
NON_RECENT_CIRCUITS = {
    "French Grand Prix",   # last held 2022
}

INDEX_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>F1 Pit-Stop Predictor</title>
<style>
  :root { --bg:#0a0e14; --panel:#141a24; --line:#2a3547; --txt:#e6edf3;
          --muted:#8b97a7; --accent:#00d2be; }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--txt);
    font-family:'Segoe UI', system-ui, sans-serif; overflow:hidden; }
  #tabs { display:flex; align-items:center; gap:2px; height:52px; padding:0 16px;
    background:var(--panel); border-bottom:1px solid var(--line); }
  #tabs .brand { font-weight:700; letter-spacing:2px; margin-right:20px; font-size:15px; }
  #tabs .brand span { color:var(--accent); }
  #tabs button { background:transparent; color:var(--muted); border:none;
    border-bottom:2px solid transparent; padding:14px 16px; font-size:14px;
    cursor:pointer; font-family:inherit; }
  #tabs button:hover { color:var(--txt); }
  #tabs button.active { color:var(--txt); border-bottom-color:var(--accent); font-weight:600; }
  .frame-wrap { position:absolute; top:52px; left:0; right:0; bottom:0; }
  iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:var(--bg); }
  iframe.hidden { visibility:hidden; pointer-events:none; }
</style>
</head>
<body>
<div id="tabs">
  <div class="brand">&#127950; F1 <span>PIT-STOP</span></div>
  <button data-tab="predictor" class="active">&#127950;&nbsp; Predictor</button>
  <button data-tab="insights">&#128202;&nbsp; Feature insights</button>
</div>
<div class="frame-wrap">
  <iframe id="predictor" src="predictor.html" title="Predictor"></iframe>
  <iframe id="insights" class="hidden" title="Feature insights"></iframe>
</div>
<script>
  const frames = {
    predictor: document.getElementById('predictor'),
    insights:  document.getElementById('insights'),
  };
  let insightsLoaded = false;
  document.getElementById('tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const tab = b.dataset.tab;
    document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('active', x === b));
    // Lazy-load the insights iframe the first time it is opened.
    if (tab === 'insights' && !insightsLoaded) { frames.insights.src = 'insights.html'; insightsLoaded = true; }
    frames.predictor.classList.toggle('hidden', tab !== 'predictor');
    frames.insights.classList.toggle('hidden', tab !== 'insights');
  });
</script>
</body>
</html>
"""


def load_json(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return json.load(f)


def write_js_asset(varname, obj, filename):
    path = os.path.join(ASSETS, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"window.{varname}=")
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";")
    return os.path.getsize(path)


def write_html(template_name, out_name, inject):
    with open(os.path.join(TPL, template_name), encoding="utf-8") as f:
        html = f.read()
    if "<!--DATA_INJECT-->" not in html:
        raise SystemExit(f"{template_name}: missing <!--DATA_INJECT--> placeholder")
    html = html.replace("<!--DATA_INJECT-->", inject)
    with open(os.path.join(OUT, out_name), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    model = load_json("model.json")
    appdata = load_json("app_data.json")
    geo = load_json("circuits_geo.json")

    # ── Filter out non-recent circuits ────────────────────────────────────────
    before = set(appdata.get("circuits", {}))
    appdata["circuits"] = {k: v for k, v in appdata.get("circuits", {}).items()
                           if k not in NON_RECENT_CIRCUITS}
    geo = {k: v for k, v in geo.items() if k not in NON_RECENT_CIRCUITS}
    removed = sorted(before - set(appdata["circuits"]))

    # ── Reset output dir ──────────────────────────────────────────────────────
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(ASSETS, exist_ok=True)

    # ── Data assets (classic scripts → set window.* before the deferred module)─
    m_kb = write_js_asset("MODEL",   model,   "model.js")     / 1024
    a_kb = write_js_asset("APPDATA", appdata, "app_data.js")  / 1024
    g_kb = write_js_asset("GEO",     geo,     "circuits_geo.js") / 1024

    # ── HTML pages ────────────────────────────────────────────────────────────
    predictor_inject = (
        '<script src="assets/model.js"></script>\n'
        '<script src="assets/circuits_geo.js"></script>\n'
        '<script src="assets/app_data.js"></script>'
    )
    insights_inject = '<script src="assets/app_data.js"></script>'
    write_html("index.html",    "predictor.html", predictor_inject)
    write_html("features.html", "insights.html",  insights_inject)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX_SHELL)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"wrote {OUT}")
    print(f"  assets/model.js        {m_kb:7.0f} KB")
    print(f"  assets/app_data.js     {a_kb:7.0f} KB")
    print(f"  assets/circuits_geo.js {g_kb:7.0f} KB")
    print(f"  index.html / predictor.html / insights.html")
    print(f"circuits: {len(appdata['circuits'])} kept, "
          f"{len(removed)} removed -> {removed}")


if __name__ == "__main__":
    main()
