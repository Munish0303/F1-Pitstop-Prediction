"""
build_circuits_geo.py
---------------------
Fetches real F1 circuit layouts (bacinger/f1-circuits GeoJSON), projects
lon/lat -> local metres (equirectangular, aspect-ratio preserved), normalises
each track to fit a fixed box, and writes static/circuits_geo.json keyed by the
Grand Prix names this project uses.

Run once (needs internet):
    python build_circuits_geo.py
"""
import json
import math
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/%s.geojson"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
OUT = os.path.join(OUT_DIR, "circuits_geo.json")

# Grand Prix name (as used by the model/aggregates) -> circuit file id
GP_TO_FILE = {
    "Abu Dhabi Grand Prix":      "ae-2009",
    "Australian Grand Prix":     "au-1953",
    "Austrian Grand Prix":       "at-1969",
    "Azerbaijan Grand Prix":     "az-2016",
    "Bahrain Grand Prix":        "bh-2002",
    "Belgian Grand Prix":        "be-1925",
    "British Grand Prix":        "gb-1948",
    "Canadian Grand Prix":       "ca-1978",
    "Chinese Grand Prix":        "cn-2004",
    "Dutch Grand Prix":          "nl-1948",
    "Emilia Romagna Grand Prix": "it-1953",
    "French Grand Prix":         "fr-1969",
    "Hungarian Grand Prix":      "hu-1986",
    "Italian Grand Prix":        "it-1922",
    "Japanese Grand Prix":       "jp-1962",
    "Las Vegas Grand Prix":      "us-2023",
    "Mexico City Grand Prix":    "mx-1962",
    "Miami Grand Prix":          "us-2022",
    "Monaco Grand Prix":         "mc-1929",
    "Qatar Grand Prix":          "qa-2004",
    "Saudi Arabian Grand Prix":  "sa-2021",
    "Singapore Grand Prix":      "sg-2008",
    "Spanish Grand Prix":        "es-1991",
    "São Paulo Grand Prix":      "br-1940",
    "United States Grand Prix":  "us-2012",
}

TARGET_SIZE = 70.0   # largest dimension of the normalised track (scene units)


def first_linestring(geo):
    """Return the longest LineString coordinate list in the GeoJSON."""
    best = []
    for feat in geo.get("features", []):
        g = feat.get("geometry", {})
        t, c = g.get("type"), g.get("coordinates")
        if t == "LineString" and len(c) > len(best):
            best = c
        elif t == "MultiLineString":
            for seg in c:
                if len(seg) > len(best):
                    best = seg
    return best


def to_local(coords):
    """lon/lat -> centred, aspect-preserving local metres, normalised to box."""
    lat0 = sum(p[1] for p in coords) / len(coords)
    k = math.cos(math.radians(lat0))
    R = 111320.0  # metres per degree latitude (good enough for a circuit)
    xs = [(p[0]) * k * R for p in coords]
    ys = [(p[1]) * R for p in coords]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    s = TARGET_SIZE / span
    # x -> scene x, lat(y) -> scene z (negate so north points "up"/-z)
    return [[round(x * s, 3), round(-y * s, 3)] for x, y in zip(xs, ys)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    for gp, fid in GP_TO_FILE.items():
        try:
            geo = json.load(urllib.request.urlopen(BASE % fid, timeout=30))
            pts = to_local(first_linestring(geo))
            out[gp] = pts
            print("ok   %-28s %-9s %4d pts" % (gp, fid, len(pts)))
        except Exception as e:
            print("FAIL %-28s %-9s %s" % (gp, fid, e))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("\nwrote %d circuits -> %s" % (len(out), OUT))


if __name__ == "__main__":
    main()
