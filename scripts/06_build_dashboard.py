"""
06_build_dashboard.py
London pedestrian road safety - Step 6: build the interactive dashboard.

Collects the results of steps 1-5, slims them down for the web, and writes a self-contained
website into docs/, ready for GitHub Pages (Settings > Pages > Deploy from branch: main, /docs).
Steps 4 and 5 are optional: their sections are left out of the dashboard if they haven't been run.

Usage (from the project folder):
    python scripts/06_build_dashboard.py
    python scripts/06_build_dashboard.py --repo https://github.com/you/london-pedestrian-safety

Outputs:
    docs/index.html     the dashboard (open it directly, or via GitHub Pages)
    docs/data/*.js      the data it reads
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
CFG = ROOT / "config"
DOCS = ROOT / "docs"
TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"

CONTEXT = OUT / "london_context.gpkg"
HOT = OUT / "london_hotspots.gpkg"
MODEL = OUT / "london_model.gpkg"
ACCESS = OUT / "london_access.gpkg"
COEFS = RES / "model_coefficients.csv"
MTC_FILE = CFG / "major_trauma_centres.csv"
AE_FILE = RES / "ae_sites_from_osm.csv"

BNG, WGS84 = "EPSG:27700", "EPSG:4326"
SIMPLIFY_M = 15
GI_CODES = {"Cold spot 99%": -3, "Cold spot 95%": -2, "Cold spot 90%": -1, "Not significant": 0,
            "Hot spot 90%": 1, "Hot spot 95%": 2, "Hot spot 99%": 3}
SEVERITY_CODES = {"Fatal": 0, "Serious": 1, "Slight": 2}


def log(msg=""):
    print(msg, flush=True)


def write_js(name, obj):
    path = DOCS / "data" / f"{name}.js"
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    path.write_text(f"window.LPS=window.LPS||{{}};LPS[{json.dumps(name)}]={payload};", encoding="utf-8")
    log(f"  data/{name}.js  {path.stat().st_size / 1e6:.2f} MB")


def clean(v):
    """JSON-safe value: NaN -> None, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def to_geojson(gdf, props):
    g = gdf.to_crs(WGS84)
    geoms = shapely.transform(g.geometry.values, lambda c: np.round(c, 5))
    feats = []
    for geom, (_, row) in zip(geoms, g[list(props)].iterrows()):
        feats.append({"type": "Feature", "geometry": shapely.geometry.mapping(geom),
                      "properties": {props[k]: clean(row[k]) for k in props}})
    return {"type": "FeatureCollection", "features": feats}


def simplify_coverage(gdf, tol):
    """Simplify shared borders together so neighbouring areas keep touching."""
    try:
        geoms = shapely.coverage_simplify(gdf.geometry.values, tol)
    except Exception:
        geoms = gdf.geometry.simplify(tol, preserve_topology=True).values
    out = gdf.copy()
    out["geometry"] = geoms
    return out


def pick_spread(df, n, gap):
    chosen = []
    for i, p in df.geometry.centroid.items():
        if all(p.distance(df.geometry.centroid[j]) >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return df.loc[chosen]


def git_remote():
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return url[:-4] if url.endswith(".git") else url or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="link to the GitHub repository, shown on the page")
    ap.add_argument("--top-streets", type=int, default=4000, help="number of hotspot street pieces to include")
    args = ap.parse_args()
    for p in (CONTEXT, HOT):
        if not p.exists():
            sys.exit(f"Missing {p}. Run steps 1-3 first.")
    if not TEMPLATE.exists():
        sys.exit(f"Missing {TEMPLATE}. Put dashboard_template.html in the scripts folder.")
    (DOCS / "data").mkdir(parents=True, exist_ok=True)

    log("== Small areas ==")
    lsoa = gpd.read_file(CONTEXT, layer="lsoa").to_crs(BNG)
    hot = gpd.read_file(HOT, layer="area_hotspots").drop(columns="geometry")
    lsoa = lsoa.merge(hot[["lsoa21cd", "gi_class", "ksi_per_road_km_yr"]], on="lsoa21cd", how="left")
    has_model = MODEL.exists()
    if has_model:
        m = gpd.read_file(MODEL, layer="lsoa_model").drop(columns="geometry")
        lsoa = lsoa.merge(m[["lsoa21cd", "observed", "expected", "ratio", "significant_fdr5"]], on="lsoa21cd", how="left")
    else:
        log("  (no step 4 results: 'more than expected' view left out)")
    has_access = ACCESS.exists()
    if has_access:
        edges = gpd.read_file(ACCESS, layer="network_time_to_trauma").to_crs(BNG)
        pts = gpd.GeoDataFrame(lsoa[["lsoa21cd"]], geometry=lsoa.geometry.representative_point(), crs=BNG)
        near = gpd.sjoin_nearest(pts, edges[["mins_to_trauma_centre", "geometry"]], how="left")
        lsoa["mins_trauma"] = near.groupby(level=0)["mins_to_trauma_centre"].min()
    else:
        log("  (no step 5 results: 'drive time to trauma care' view left out)")
    lsoa["gi"] = lsoa["gi_class"].map(GI_CODES).fillna(0).astype(int)
    lsoa = simplify_coverage(lsoa, SIMPLIFY_M)
    props = {"lsoa21nm": "n", "borough": "b", "casualties_per_km2_yr": "d", "ksi_per_road_km_yr": "k",
             "n_casualties": "c", "n_ksi": "s", "gi": "g"}
    if has_model:
        props.update({"observed": "o", "expected": "e", "ratio": "r", "significant_fdr5": "x"})
    if has_access:
        props["mins_trauma"] = "t"
    write_js("lsoa", to_geojson(lsoa, props))

    log("== Boroughs ==")
    rates = gpd.read_file(HOT, layer="borough_rates").drop(columns="geometry")
    outline = lsoa.dissolve("borough").reset_index()[["borough", "geometry"]]
    b = outline.merge(rates, on="borough", how="left")
    bprops = {"borough": "name", "population": "pop", "n_ksi": "ksi", "n_casualties": "all",
              "ksi_per_100k_pop_yr": "ksi_pop", "ksi_per_road_km_yr": "ksi_road", "ksi_per_km2_yr": "ksi_area",
              "rank_per_resident": "r_pop", "rank_per_road_km": "r_road", "rank_per_km2": "r_area"}
    write_js("boroughs", to_geojson(b, {k: v for k, v in bprops.items() if k in b.columns}))
    boroughs = sorted(b["borough"].dropna().unique())

    log("== Casualties ==")
    if has_access:
        cas = gpd.read_file(ACCESS, layer="casualties_access")
    else:
        cas = gpd.read_file(CONTEXT, layer="casualties")
    cas = cas.to_crs(WGS84)
    cas = cas[cas.geometry.notna()]
    age = pd.to_numeric(cas.get("age_of_casualty"), errors="coerce") if "age_of_casualty" in cas.columns else None
    bidx = {n: i for i, n in enumerate(boroughs)}
    columnar = {
        "lat": np.round(cas.geometry.y, 5).tolist(),
        "lon": np.round(cas.geometry.x, 5).tolist(),
        "year": pd.to_numeric(cas["year"], errors="coerce").fillna(0).astype(int).tolist(),
        "sev": cas["severity"].map(SEVERITY_CODES).fillna(2).astype(int).tolist(),
        "hour": pd.to_numeric(cas.get("hour"), errors="coerce").fillna(-1).astype(int).tolist()
        if "hour" in cas.columns else [-1] * len(cas),
        "age": (age.where(age >= 0).fillna(-1).astype(int).tolist() if age is not None else [-1] * len(cas)),
        "b": cas["borough"].map(bidx).fillna(-1).astype(int).tolist() if "borough" in cas.columns else [-1] * len(cas),
    }
    if has_access:
        columnar["mt"] = [clean(v) for v in cas["mins_to_trauma_centre"].round(1)]
        columnar["ae"] = [clean(v) for v in cas["mins_to_ae"].round(1)]
    write_js("cas", columnar)

    log("== Hotspot streets ==")
    st = gpd.read_file(HOT, layer="street_hotspots")
    st = st[st["ksi_density"] > 0].sort_values("ksi_density", ascending=False)
    top = st.head(args.top_streets).to_crs(WGS84)
    streets = [{"c": [[round(y, 5), round(x, 5)] for x, y in g.coords], "d": round(float(d), 3),
                "n": n if isinstance(n, str) else None}
               for g, d, n in zip(top.geometry, top["ksi_density"], top["name"])]
    write_js("streets", streets)
    named = st[st["name"].apply(lambda v: isinstance(v, str) and v != "")].head(20000)
    named = gpd.sjoin(gpd.GeoDataFrame(named[["name", "ksi_density"]], geometry=named.geometry.centroid, crs=BNG),
                      outline.to_crs(BNG), predicate="within")
    top_by_b = {}
    for bn, grp in named.groupby("borough"):
        picks = pick_spread(grp.sort_values("ksi_density", ascending=False), 3, 500).to_crs(WGS84)
        top_by_b[bn] = [{"n": r["name"], "d": round(float(r["ksi_density"]), 2),
                         "ll": [round(r.geometry.y, 5), round(r.geometry.x, 5)]} for _, r in picks.iterrows()]
    write_js("topstreets", top_by_b)

    model = []
    if COEFS.exists():
        c = pd.read_csv(COEFS)
        c = c[c["model"].str.startswith("B") & (c["subset"] == "All London")]
        model = [{k: clean(r[k]) for k in ("outcome", "label", "group", "pct_change", "pct_low", "pct_high", "p_value")}
                 for _, r in c.iterrows()]
    write_js("model", model)

    hosp = {"mtc": [], "ae": []}
    if MTC_FILE.exists():
        hosp["mtc"] = pd.read_csv(MTC_FILE)[["name", "lat", "lon"]].to_dict("records")
    if AE_FILE.exists():
        names = {h["name"] for h in hosp["mtc"]}
        hosp["ae"] = [r for r in pd.read_csv(AE_FILE)[["name", "lat", "lon"]].to_dict("records") if r["name"] not in names]
    write_js("hospitals", hosp)

    years = sorted(set(columnar["year"]) - {0})
    meta = {"built": datetime.date.today().strftime("%d %B %Y"), "years": years, "boroughs": boroughs,
            "repo": args.repo or git_remote(), "has_model": has_model and bool(model), "has_access": has_access,
            "n_lsoa": int(len(lsoa))}
    write_js("meta", meta)

    shutil.copy2(TEMPLATE, DOCS / "index.html")
    (DOCS / ".nojekyll").write_text("")   # tells GitHub Pages to serve the files as they are
    log(f"\nDashboard written to {DOCS / 'index.html'}")
    log("Open it in your browser to check, then push and switch on GitHub Pages (Settings > Pages > main, /docs).")


if __name__ == "__main__":
    main()
