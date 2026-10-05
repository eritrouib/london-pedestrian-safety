"""
02_build_context.py
London pedestrian road safety - Step 2: the context around each casualty.

Adds small-area boundaries, population, deprivation, the built environment from
OpenStreetMap and the street network, then links every pedestrian casualty from
step 1 to its neighbourhood and nearby street features.

Usage (from the project folder, after step 1):
    python scripts/02_build_context.py
    python scripts/02_build_context.py --skip-network   # faster: no street network download
    python scripts/02_build_context.py --refresh        # re-download everything

Sources (all Open Government Licence or ODbL):
    Lower layer Super Output Areas (Dec 2021) boundaries   ONS Open Geography Portal
    English Indices of Deprivation 2025, File 7            MHCLG (scores, deciles, population mid-2022)
    OpenStreetMap, Greater London extract                  Geofabrik (© OpenStreetMap contributors, ODbL)
    Drivable street network                                OpenStreetMap via OSMnx

Outputs (data/processed/):
    london_context.gpkg
        lsoa          every London small area with population, deprivation, built environment and casualty counts
        boroughs      the same, summed up to the 33 boroughs
        casualties    step 1 casualties + small area + distance to the nearest street features + nearest road type
        osm_features  crossings, signals, bus stops, stations, schools, pubs/bars
        roads         OSM roads by class
    lsoa_variables.csv   the lsoa table without geometry, ready for modelling (step 4)
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import requests
import shapely

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
STEP1_GPKG = OUT / "london_pedestrian_safety.gpkg"

IOD_URL = ("https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/"
           "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv")
LSOA_QUERY = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
              "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query")
OSM_URL = "https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf"

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

ROAD_GROUPS = {
    "major": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"],
    "secondary": ["secondary", "secondary_link", "tertiary", "tertiary_link"],
    "local": ["unclassified", "residential", "living_street"],
}
HIGHWAY_TO_GROUP = {h: g for g, hs in ROAD_GROUPS.items() for h in hs}

FEATURE_TYPES = ["crossing", "traffic_signals", "bus_stop", "station", "school", "pub_bar"]


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)


def download(url, dest, refresh=False):
    if dest.exists() and not refresh:
        log(f"  already have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"  downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def find_col(df, *keys, required=True):
    """First column whose name contains all the given fragments (case-insensitive)."""
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keys):
            return c
    if required:
        sys.exit(f"!! No column containing {keys}. Columns: {list(df.columns)[:12]} ...")
    return None


def per_km2(n, area_km2):
    return np.where(area_km2 > 0, n / area_km2, np.nan)


def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


# --------------------------------------------------------------------------
# 1. DEPRIVATION + POPULATION
# --------------------------------------------------------------------------
def load_iod(refresh):
    path = download(IOD_URL, RAW / "iod2025_file7.csv", refresh)
    df = pd.read_csv(path)
    c = dict(
        code=find_col(df, "LSOA code"),
        name=find_col(df, "LSOA name"),
        lad=find_col(df, "Local Authority District code"),
        lad_name=find_col(df, "Local Authority District name"),
        imd_score=find_col(df, "Index of Multiple Deprivation", "Score"),
        imd_decile=find_col(df, "Index of Multiple Deprivation", "Decile"),
        income_score=find_col(df, "Income Score", required=False),
        employment_score=find_col(df, "Employment Score", required=False),
    )
    pop = next((col for col in df.columns if col.lower().startswith("total population")), None)
    if not pop:
        sys.exit("!! No 'Total population' column in the IoD file.")
    c["population"] = pop
    out = df[[v for v in c.values() if v]].rename(columns={v: k for k, v in c.items() if v})
    out = out[out["lad"].astype(str).str.startswith("E09")].copy()
    log(f"  {len(out):,} London small areas; population column: '{pop}'")
    return out


# --------------------------------------------------------------------------
# 2. BOUNDARIES
# --------------------------------------------------------------------------
def load_lsoa_boundaries(codes, refresh):
    cache = RAW / "london_lsoa_2021.gpkg"
    if cache.exists() and not refresh:
        log(f"  already have {cache.name}")
        return gpd.read_file(cache)
    parts = []
    codes = list(codes)
    for i in range(0, len(codes), 200):
        batch = codes[i:i + 200]
        where = "LSOA21CD IN (" + ",".join(f"'{c}'" for c in batch) + ")"
        r = requests.post(LSOA_QUERY, timeout=180, data=dict(
            where=where, outFields="LSOA21CD,LSOA21NM", outSR=4326, f="geojson", returnGeometry="true"))
        r.raise_for_status()
        gj = r.json()
        if "error" in gj:
            sys.exit(f"!! ONS boundary service error: {gj['error']}")
        parts.append(gpd.GeoDataFrame.from_features(gj["features"], crs=WGS84))
        print(f"\r    boundaries {min(i + 200, len(codes)):,} / {len(codes):,}", end="", flush=True)
    print()
    g = pd.concat(parts, ignore_index=True).to_crs(BNG)
    g = g.rename(columns={"LSOA21CD": "lsoa21cd", "LSOA21NM": "lsoa21nm"})[["lsoa21cd", "lsoa21nm", "geometry"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(cache, driver="GPKG")
    return g


# --------------------------------------------------------------------------
# 3. OPENSTREETMAP: features + roads
# --------------------------------------------------------------------------
def _tag(other_tags, key):
    if not isinstance(other_tags, str):
        return None
    m = re.search(r'"%s"=>"([^"]*)"' % re.escape(key), other_tags)
    return m.group(1) if m else None


def classify_point(highway, other_tags):
    if highway in ("crossing", "traffic_signals", "bus_stop"):
        return highway
    amenity = _tag(other_tags, "amenity")
    if amenity == "school":
        return "school"
    if amenity in ("pub", "bar"):
        return "pub_bar"
    if _tag(other_tags, "railway") == "station" or _tag(other_tags, "public_transport") == "station":
        return "station"
    return None


def load_osm(osm_file, boundary):
    log(f"  reading {osm_file.name} (this can take a few minutes) ...")
    t0 = time.time()
    pts = pyogrio.read_dataframe(
        osm_file, layer="points", columns=["osm_id", "name", "highway", "other_tags"],
        where=("highway IN ('crossing','traffic_signals','bus_stop') "
               "OR other_tags LIKE '%\"amenity\"=>\"school\"%' OR other_tags LIKE '%\"amenity\"=>\"pub\"%' "
               "OR other_tags LIKE '%\"amenity\"=>\"bar\"%' OR other_tags LIKE '%\"railway\"=>\"station\"%' "
               "OR other_tags LIKE '%\"public_transport\"=>\"station\"%'"))
    pts["feature"] = [classify_point(h, o) for h, o in zip(pts["highway"], pts["other_tags"])]

    # schools and pubs are often mapped as building outlines rather than points
    polys = pyogrio.read_dataframe(
        osm_file, layer="multipolygons", columns=["osm_id", "osm_way_id", "name", "amenity"],
        where="amenity IN ('school','pub','bar')")
    polys["feature"] = polys["amenity"].map({"school": "school", "pub": "pub_bar", "bar": "pub_bar"})
    polys = polys.to_crs(BNG)
    polys["geometry"] = polys.geometry.representative_point()

    feats = pd.concat([pts.to_crs(BNG)[["name", "feature", "geometry"]],
                       polys[["name", "feature", "geometry"]]], ignore_index=True)
    feats = gpd.GeoDataFrame(feats[feats["feature"].notna()], geometry="geometry", crs=BNG)

    roads = pyogrio.read_dataframe(
        osm_file, layer="lines", columns=["osm_id", "name", "highway"],
        where="highway IN (" + ",".join(f"'{h}'" for h in HIGHWAY_TO_GROUP) + ")").to_crs(BNG)
    roads["road_group"] = roads["highway"].map(HIGHWAY_TO_GROUP)

    # keep what lies in London
    feats = feats[feats.intersects(boundary)].reset_index(drop=True)
    roads = roads[roads.intersects(boundary)].reset_index(drop=True)
    log(f"  {len(feats):,} street features and {len(roads):,} road segments in {time.time() - t0:.0f}s")
    log("  " + ", ".join(f"{k}: {v:,}" for k, v in feats["feature"].value_counts().items()))
    return feats, roads


def road_length_by_lsoa(roads, lsoa):
    pairs = gpd.sjoin(roads[["road_group", "geometry"]], lsoa[["lsoa21cd", "geometry"]],
                      predicate="intersects", how="inner")
    shapes = lsoa.set_index("lsoa21cd").geometry
    pieces = shapely.intersection(pairs.geometry.values, shapes.loc[pairs["lsoa21cd"]].values)
    pairs["km"] = shapely.length(pieces) / 1000
    return pairs.pivot_table(index="lsoa21cd", columns="road_group", values="km", aggfunc="sum", fill_value=0)


# --------------------------------------------------------------------------
# 4. STREET NETWORK (junctions)
# --------------------------------------------------------------------------
def load_junctions(boundary, refresh):
    try:
        import osmnx as ox
    except ImportError:
        log("  ! osmnx not installed (pip install osmnx); skipping junctions")
        return None
    # keep OSMnx's raw download cache inside data/ (ignored by git) instead of ./cache
    ox.settings.cache_folder = str(RAW / "osm" / "cache")
    cache = RAW / "osm" / "london_drive.graphml"
    if cache.exists() and not refresh:
        log(f"  loading cached {cache.name}")
        G = ox.load_graphml(cache)
    else:
        log("  downloading the drivable street network from OpenStreetMap (10-20 minutes) ...")
        poly = gpd.GeoSeries([boundary], crs=BNG).to_crs(WGS84).iloc[0]
        G = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, cache)
    nodes = ox.graph_to_gdfs(G, edges=False)
    if "street_count" not in nodes.columns:   # not stored in every saved network
        nodes["street_count"] = pd.Series(ox.stats.count_streets_per_node(G))
    nodes["street_count"] = pd.to_numeric(nodes["street_count"], errors="coerce")
    junctions = nodes[nodes["street_count"] >= 3].to_crs(BNG)[["street_count", "geometry"]]
    log(f"  {len(junctions):,} junctions (3 or more streets meeting)")
    return junctions.reset_index(drop=True)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download all sources")
    ap.add_argument("--skip-network", action="store_true", help="skip the street network (no junction density)")
    ap.add_argument("--osm-file", type=Path, help="use this OSM extract instead of downloading Greater London")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not STEP1_GPKG.exists():
        sys.exit(f"Missing {STEP1_GPKG}. Run scripts/01_get_stats19.py first.")

    log("== 1. Deprivation and population (IoD 2025) ==")
    iod = load_iod(args.refresh)

    log("\n== 2. Small-area boundaries (LSOA 2021) ==")
    shapes = load_lsoa_boundaries(iod["code"], args.refresh)
    lsoa = shapes.merge(iod.drop(columns=["name"]), left_on="lsoa21cd", right_on="code", how="inner").drop(columns="code")
    lsoa = lsoa.rename(columns={"lad_name": "borough"})
    missing = len(iod) - len(lsoa)
    log(f"  {len(lsoa):,} small areas with boundaries" + (f" ({missing} without, skipped)" if missing else ""))
    lsoa["area_km2"] = lsoa.area / 1e6
    lsoa["pop_density"] = per_km2(lsoa["population"], lsoa["area_km2"])
    boundary = shapely.union_all(lsoa.geometry.values)

    log("\n== 3. Built environment (OpenStreetMap) ==")
    osm_file = args.osm_file or download(OSM_URL, RAW / "osm" / "greater-london-latest.osm.pbf", args.refresh)
    feats, roads = load_osm(osm_file, boundary)

    counts = gpd.sjoin(feats, lsoa[["lsoa21cd", "geometry"]], predicate="within") \
        .groupby(["lsoa21cd", "feature"]).size().unstack(fill_value=0)
    for f in FEATURE_TYPES:
        n = counts[f] if f in counts.columns else pd.Series(dtype=float)
        lsoa[f"n_{f}"] = lsoa["lsoa21cd"].map(n).fillna(0).astype(int)
        lsoa[f"{f}_per_km2"] = per_km2(lsoa[f"n_{f}"], lsoa["area_km2"])

    lengths = road_length_by_lsoa(roads, lsoa)
    for g in ROAD_GROUPS:
        km = lsoa["lsoa21cd"].map(lengths[g] if g in lengths.columns else pd.Series(dtype=float)).fillna(0)
        lsoa[f"road_km_{g}"] = km.round(3)
        lsoa[f"road_density_{g}"] = per_km2(km, lsoa["area_km2"])
    lsoa["road_density_all"] = per_km2(lsoa[[f"road_km_{g}" for g in ROAD_GROUPS]].sum(axis=1), lsoa["area_km2"])
    lsoa["major_road_share"] = np.where(lsoa["road_density_all"] > 0,
                                        lsoa["road_density_major"] / lsoa["road_density_all"], np.nan)

    log("\n== 4. Street network ==")
    junctions = None if args.skip_network else load_junctions(boundary, args.refresh)
    if junctions is not None:
        jc = gpd.sjoin(junctions, lsoa[["lsoa21cd", "geometry"]], predicate="within").groupby("lsoa21cd").size()
        lsoa["n_junctions"] = lsoa["lsoa21cd"].map(jc).fillna(0).astype(int)
        lsoa["junction_density"] = per_km2(lsoa["n_junctions"], lsoa["area_km2"])
    else:
        log("  skipped (no junction density this run)")

    log("\n== 5. Linking casualties ==")
    cas = gpd.read_file(STEP1_GPKG, layer="casualties").to_crs(BNG)
    cas = gpd.sjoin(cas, lsoa[["lsoa21cd", "geometry"]], predicate="within", how="left").drop(columns="index_right")
    log(f"  {cas['lsoa21cd'].notna().sum():,} of {len(cas):,} casualties placed in a London small area")

    for f in FEATURE_TYPES:
        sub = feats[feats["feature"] == f][["geometry"]]
        if len(sub):
            near = gpd.sjoin_nearest(cas[["geometry"]], sub, distance_col="d", how="left")
            cas[f"dist_{f}_m"] = near.groupby(level=0)["d"].min().round(1)
    near_road = gpd.sjoin_nearest(cas[["geometry"]], roads[["highway", "road_group", "geometry"]],
                                  how="left", max_distance=50, distance_col="d")
    near_road = near_road.sort_values("d").groupby(level=0).first()
    cas["nearest_road_type"] = near_road["highway"]
    cas["nearest_road_group"] = near_road["road_group"]
    if junctions is not None:
        nj = gpd.sjoin_nearest(cas[["geometry"]], junctions[["geometry"]], distance_col="d", how="left")
        cas["dist_junction_m"] = nj.groupby(level=0)["d"].min().round(1)

    # casualty counts per small area
    years = cas["year"].nunique() if "year" in cas.columns else 5
    ksi = cas["severity"].isin(["Fatal", "Serious"])
    lsoa["n_casualties"] = lsoa["lsoa21cd"].map(cas.groupby("lsoa21cd").size()).fillna(0).astype(int)
    lsoa["n_ksi"] = lsoa["lsoa21cd"].map(cas[ksi].groupby("lsoa21cd").size()).fillna(0).astype(int)
    lsoa["casualties_per_km2_yr"] = per_km2(lsoa["n_casualties"], lsoa["area_km2"]) / years
    lsoa["ksi_per_1000_pop_yr"] = np.where(lsoa["population"] > 0,
                                           1000 * lsoa["n_ksi"] / lsoa["population"] / years, np.nan)

    log("\n== 6. Boroughs ==")
    sum_cols = ["area_km2", "population", "n_casualties", "n_ksi"] + [f"n_{f}" for f in FEATURE_TYPES] \
        + [f"road_km_{g}" for g in ROAD_GROUPS] + (["n_junctions"] if "n_junctions" in lsoa.columns else [])
    lsoa["_imd_x_pop"] = lsoa["imd_score"] * lsoa["population"]
    boroughs = lsoa.dissolve(by=["lad", "borough"], aggfunc={**{c: "sum" for c in sum_cols}, "_imd_x_pop": "sum"}).reset_index()
    boroughs["imd_score_popweighted"] = boroughs["_imd_x_pop"] / boroughs["population"]
    boroughs = boroughs.drop(columns="_imd_x_pop")
    lsoa = lsoa.drop(columns="_imd_x_pop")
    boroughs["ksi_per_100k_pop_yr"] = 1e5 * boroughs["n_ksi"] / boroughs["population"] / years
    boroughs["ksi_per_km2_yr"] = boroughs["n_ksi"] / boroughs["area_km2"] / years
    log(f"  {len(boroughs)} boroughs")

    log("\n== 7. Saving ==")
    gpkg = OUT / "london_context.gpkg"
    num = lsoa.select_dtypes("number").columns
    lsoa[num] = lsoa[num].round(4)
    lsoa.to_file(gpkg, layer="lsoa", driver="GPKG")
    boroughs.to_file(gpkg, layer="boroughs", driver="GPKG")
    cas.to_file(gpkg, layer="casualties", driver="GPKG")
    feats.to_file(gpkg, layer="osm_features", driver="GPKG")
    roads[["name", "highway", "road_group", "geometry"]].to_file(gpkg, layer="roads", driver="GPKG")
    lsoa.drop(columns="geometry").to_csv(OUT / "lsoa_variables.csv", index=False)
    log(f"  {gpkg.name}: lsoa, boroughs, casualties, osm_features, roads")
    log("  lsoa_variables.csv")

    log("\n== Summary ==")
    b = boroughs.set_index("borough")
    log("Fatal + serious pedestrian casualties per 100,000 residents per year (top 10):")
    log(b["ksi_per_100k_pop_yr"].sort_values(ascending=False).head(10).round(1).to_string())
    log("\nNote: rates per resident overstate central areas such as the City of London and Westminster,")
    log("where far more people walk than live. Step 3 adds per-area density and hotspot analysis.")

    log("\nFirst look: how small-area variables relate to fatal + serious casualties per km²")
    log("(Spearman rank correlation, 1 = rises together, -1 = opposite, 0 = no relation)")
    target = per_km2(lsoa["n_ksi"], lsoa["area_km2"])
    target = pd.Series(target, index=lsoa.index)
    candidates = ["pop_density", "imd_score", "crossing_per_km2", "traffic_signals_per_km2", "bus_stop_per_km2",
                  "station_per_km2", "school_per_km2", "pub_bar_per_km2", "road_density_major",
                  "major_road_share", "junction_density"]
    rows = [(c, spearman(lsoa[c], target)) for c in candidates if c in lsoa.columns]
    for c, r in sorted(rows, key=lambda x: -abs(x[1]) if pd.notna(x[1]) else 0):
        log(f"  {c:<26} {r:+.2f}")
    log("\nThese are associations, not causes: busy places have more of everything, including people on foot.")
    log("Done. Next: step 3 finds hotspots and builds fair rates.")


if __name__ == "__main__":
    main()
