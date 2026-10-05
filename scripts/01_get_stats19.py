"""
01_get_stats19.py
London pedestrian road safety - Step 1: get the collision data.

Downloads the Department for Transport's STATS19 road safety open data
(police-recorded injury collisions, Great Britain), keeps pedestrian casualties
in the chosen area, and saves analysis-ready spatial data.

Usage (from the project folder):
    python scripts/01_get_stats19.py                 # London, last 5 published years
    python scripts/01_get_stats19.py --area england  # all of England
    python scripts/01_get_stats19.py --refresh       # re-download even if files exist
    python scripts/01_get_stats19.py --no-download   # use files already in data/raw

Outputs (data/processed/):
    <area>_pedestrian_safety.gpkg
        layer "casualties"  one row per pedestrian casualty, with its collision's details
        layer "collisions"  one row per collision that injured at least one pedestrian
    <area>_pedestrian_casualties.parquet   same as the casualties layer (GeoParquet)
    columns_report.csv                     every column found, how complete it is, example values
    <area>_quicklook_map.html              fatal and serious casualties on an interactive map

Data: Department for Transport, Road Safety Data, Open Government Licence v3.0.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data/"
FILES = {
    "collision": "dft-road-casualty-statistics-collision-last-5-years.csv",
    "casualty": "dft-road-casualty-statistics-casualty-last-5-years.csv",
}
# Code lookups for every variable; used in step 2 to decode values into labels
DATA_GUIDE_URL = ("https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/"
                  "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

LONDON_POLICE_FORCES = {1: "Metropolitan Police", 48: "City of London Police"}
SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
PEDESTRIAN_CLASS = 3          # casualty_class: 1 driver/rider, 2 passenger, 3 pedestrian

LONDON_BOROUGHS = {
    "E09000001": "City of London", "E09000002": "Barking and Dagenham", "E09000003": "Barnet",
    "E09000004": "Bexley", "E09000005": "Brent", "E09000006": "Bromley", "E09000007": "Camden",
    "E09000008": "Croydon", "E09000009": "Ealing", "E09000010": "Enfield", "E09000011": "Greenwich",
    "E09000012": "Hackney", "E09000013": "Hammersmith and Fulham", "E09000014": "Haringey",
    "E09000015": "Harrow", "E09000016": "Havering", "E09000017": "Hillingdon", "E09000018": "Hounslow",
    "E09000019": "Islington", "E09000020": "Kensington and Chelsea", "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth", "E09000023": "Lewisham", "E09000024": "Merton", "E09000025": "Newham",
    "E09000026": "Redbridge", "E09000027": "Richmond upon Thames", "E09000028": "Southwark",
    "E09000029": "Sutton", "E09000030": "Tower Hamlets", "E09000031": "Waltham Forest",
    "E09000032": "Wandsworth", "E09000033": "Westminster",
}

# Columns read as text so codes keep their leading zeros
TEXT_HINTS = ("index", "reference", "lsoa", "ons", "highway", "district")


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
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    log(f"  saved in {time.time() - t0:.0f}s")
    return dest


def read_stats19(path):
    """Read a STATS19 CSV with consistent column names across data specifications."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: str for c in header if any(h in c.lower() for h in TEXT_HINTS)}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False, na_values=["NULL", "", " "])
    df.columns = [c.strip().lower() for c in df.columns]
    # older releases say "accident", newer ones "collision"
    df = df.rename(columns={c: c.replace("accident", "collision") for c in df.columns if "accident" in c})
    return df


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def columns_report(df, name):
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(dict(table=name, column=c, non_null_pct=round(100 * s.notna().mean(), 1),
                         distinct=s.nunique(dropna=True),
                         examples=", ".join(map(str, s.dropna().unique()[:5]))))
    return pd.DataFrame(rows)


def require(df, cols, table):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log(f"\n!! The {table} file has no column(s) {missing}.")
        log(f"   Columns found: {list(df.columns)}")
        log("   DfT may have renamed them; check the data guide in data/raw and update this script.")
        sys.exit(1)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", choices=["london", "england", "gb"], default="london")
    ap.add_argument("--refresh", action="store_true", help="re-download the source files")
    ap.add_argument("--no-download", action="store_true", help="only use files already in data/raw")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("== 1. Source files ==")
    paths = {k: RAW / f for k, f in FILES.items()}
    if not args.no_download:
        for k, f in FILES.items():
            download(BASE_URL + f, paths[k], args.refresh)
        try:
            download(DATA_GUIDE_URL, RAW / "stats19_data_guide.xlsx", args.refresh)
        except Exception as e:
            log(f"  (data guide not downloaded: {e})")
    for p in paths.values():
        if not p.exists():
            sys.exit(f"Missing {p}. Run without --no-download first.")

    log("\n== 2. Reading ==")
    coll = read_stats19(paths["collision"])
    cas = read_stats19(paths["casualty"])
    log(f"  collisions: {len(coll):,} rows, {coll.shape[1]} columns")
    log(f"  casualties: {len(cas):,} rows, {cas.shape[1]} columns")
    require(coll, ["collision_index"], "collision")
    require(cas, ["collision_index", "casualty_class", "casualty_severity"], "casualty")
    pd.concat([columns_report(coll, "collision"), columns_report(cas, "casualty")]) \
        .to_csv(OUT / "columns_report.csv", index=False)
    log(f"  column check written to {OUT / 'columns_report.csv'}")

    log("\n== 3. Pedestrian casualties ==")
    cas["casualty_class"] = to_num(cas["casualty_class"])
    ped = cas[cas["casualty_class"] == PEDESTRIAN_CLASS].copy()
    log(f"  {len(ped):,} pedestrian casualties in Great Britain")
    if "casualty_type" in ped.columns:
        agree = (to_num(ped["casualty_type"]) == 0).mean()
        log(f"  check: {agree:.1%} also have casualty_type = pedestrian")

    # one collision row per pedestrian casualty; collision columns win on name clashes
    ped = ped.merge(coll, on="collision_index", how="left", suffixes=("_cas", ""))
    unmatched = ped["date"].isna().sum() if "date" in ped.columns else 0
    if unmatched:
        log(f"  ! {unmatched:,} casualties had no matching collision record")

    log(f"\n== 4. Area: {args.area} ==")
    ons = next((c for c in ("local_authority_ons_district", "local_authority_ons_code") if c in ped.columns), None)
    if args.area == "london":
        keep = pd.Series(False, index=ped.index)
        if "police_force" in ped.columns:
            keep |= to_num(ped["police_force"]).isin(LONDON_POLICE_FORCES.keys())
        if ons:
            keep |= ped[ons].fillna("").str.startswith("E09")
        ped = ped[keep]
    elif args.area == "england":
        if not ons:
            sys.exit("No ONS district column to filter England; try --area gb")
        ped = ped[ped[ons].fillna("").str.startswith("E")]
    log(f"  {len(ped):,} pedestrian casualties kept")

    log("\n== 5. Cleaning ==")
    ped["severity"] = to_num(ped["casualty_severity"]).map(SEVERITY)
    if "date" in ped.columns:
        ped["date"] = pd.to_datetime(ped["date"], format="%d/%m/%Y", errors="coerce")
        ped["year"] = ped["date"].dt.year
        ped["month"] = ped["date"].dt.month
    if "time" in ped.columns:
        ped["hour"] = pd.to_datetime(ped["time"], format="%H:%M", errors="coerce").dt.hour
    if ons:
        ped["borough"] = ped[ons].map(LONDON_BOROUGHS) if args.area == "london" else None
    adjusted = [c for c in ped.columns if "adjusted" in c and "serious" in c]
    if adjusted:
        ped[adjusted[0]] = to_num(ped[adjusted[0]])
        log(f"  severity-adjusted field found: {adjusted[0]} (use it for multi-year comparisons)")
    # STATS19 codes missing values as -1 (coordinates excluded: -1 is a real longitude)
    for c in ped.columns:
        if ped[c].dtype.kind in "if" and c not in ("longitude", "latitude"):
            ped[c] = ped[c].where(ped[c] != -1)

    log("\n== 6. Locations ==")
    nan = pd.Series(float("nan"), index=ped.index)
    e = to_num(ped["location_easting_osgr"]) if "location_easting_osgr" in ped.columns else nan
    n = to_num(ped["location_northing_osgr"]) if "location_northing_osgr" in ped.columns else nan
    has_bng = e.notna() & n.notna() & (e > 0) & (n > 0)
    gdf = gpd.GeoDataFrame(ped[has_bng], geometry=gpd.points_from_xy(e[has_bng], n[has_bng]), crs=BNG)
    rest = ped[~has_bng]
    if len(rest) and {"longitude", "latitude"} <= set(rest.columns):
        lon, lat = to_num(rest["longitude"]), to_num(rest["latitude"])
        ok = lon.notna() & lat.notna()
        if ok.any():
            extra = gpd.GeoDataFrame(rest[ok], geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs=WGS84).to_crs(BNG)
            gdf = pd.concat([gdf, extra])
    dropped = len(ped) - len(gdf)
    log(f"  {len(gdf):,} located, {dropped:,} without usable coordinates dropped")

    # one row per collision that injured at least one pedestrian
    sev_rank = to_num(gdf["casualty_severity"])
    collisions = (gdf.assign(_sev=sev_rank)
                  .sort_values("_sev")
                  .groupby("collision_index", as_index=False)
                  .agg(pedestrian_casualties=("collision_index", "size"), worst_severity=("severity", "first"),
                       **{c: (c, "first") for c in gdf.columns
                          if c not in ("collision_index", "severity", "_sev") and not c.endswith("_cas")
                          and c not in cas.columns}))
    collisions = gpd.GeoDataFrame(collisions, geometry="geometry", crs=BNG)

    log("\n== 7. Saving ==")
    gpkg = OUT / f"{args.area}_pedestrian_safety.gpkg"
    # GeoPackage cannot store pandas' nullable types well; plain objects are safest
    def _ready(g):
        g = g.copy()
        for c in g.columns:
            if c != "geometry" and str(g[c].dtype) in ("Int64", "Float64", "boolean"):
                g[c] = g[c].astype("float64" if g[c].dtype.kind in "if" else "object")
        return g
    _ready(gdf).to_file(gpkg, layer="casualties", driver="GPKG")
    _ready(collisions).to_file(gpkg, layer="collisions", driver="GPKG")
    log(f"  {gpkg.name}: casualties ({len(gdf):,}), collisions ({len(collisions):,})")
    try:
        _ready(gdf).to_parquet(OUT / f"{args.area}_pedestrian_casualties.parquet")
        log(f"  {args.area}_pedestrian_casualties.parquet")
    except Exception as ex:
        log(f"  (parquet skipped: {ex}; pip install pyarrow)")

    try:
        ks = gdf[gdf["severity"].isin(["Fatal", "Serious"])]
        cols = [c for c in ("severity", "date", "time", "borough", "speed_limit", "geometry") if c in ks.columns]
        m = ks[cols].to_crs(WGS84).explore(column="severity", cmap=["#B8403A", "#E8A33A"],
                                           tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                                                  "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
                                           attr="Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors",
                                           max_zoom=16, marker_kwds={"radius": 3},
                                           legend=True)
        m.save(OUT / f"{args.area}_quicklook_map.html")
        log(f"  {args.area}_quicklook_map.html ({len(ks):,} fatal and serious casualties)")
    except Exception as ex:
        log(f"  (quick-look map skipped: {ex}; pip install folium mapclassify matplotlib)")

    log("\n== Summary ==")
    if "year" in gdf.columns:
        t = pd.crosstab(gdf["year"], gdf["severity"]).reindex(columns=["Fatal", "Serious", "Slight"], fill_value=0)
        t["Total"] = t.sum(axis=1)
        log(t.to_string())
    if args.area == "london" and "borough" in gdf.columns:
        top = gdf[gdf["severity"].isin(["Fatal", "Serious"])].groupby("borough").size().sort_values(ascending=False)
        log("\nBoroughs with most fatal or serious pedestrian casualties:")
        log(top.head(10).to_string())
    log("\nDone. Next: step 2 adds boundaries, population and the built environment.")


if __name__ == "__main__":
    main()
