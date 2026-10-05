"""
05_access_to_care.py
London pedestrian road safety - Step 5: how far is each casualty from emergency care?

For every pedestrian casualty, the drive time along the street network to:
  - the nearest A&E department
  - the nearest of London's four major trauma centres (where the most seriously injured are taken)

Hospitals:
  - Major trauma centres: The Royal London, St Mary's, King's College and St George's, located from
    their postcodes (postcodes.io). Stored in config/major_trauma_centres.csv.
  - A&E departments: hospitals tagged with an emergency department in OpenStreetMap
    (amenity=hospital + emergency=yes), plus the trauma centres. The list found is saved to
    results/ae_sites_from_osm.csv. To correct it, copy it to config/ae_sites.csv, edit, and re-run:
    the script then uses your list instead.

Drive times use speed limits from OpenStreetMap (typical speeds by road type where missing) with no
traffic. They describe the hospital journey from the scene under free-flowing conditions, not
ambulance response times, which also depend on where ambulances are when called.

Usage (from the project folder, after steps 1-2 with the street network):
    python scripts/05_access_to_care.py

Outputs:
    data/processed/london_access.gpkg     casualties with drive times; network coloured by time to trauma care
    results/access_by_borough.csv         median and share of serious casualties beyond 15 / 20 minutes
    results/ae_sites_from_osm.csv         the A&E list used (review it)
    figures/access_to_trauma_care.png     drive time to the nearest major trauma centre, with serious casualties
    figures/access_curve.png              share of serious casualties within X minutes of care
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import requests
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
CFG = ROOT / "config"
CONTEXT = OUT / "london_context.gpkg"
GRAPH = RAW / "osm" / "london_drive.graphml"
OSM_FILE = RAW / "osm" / "greater-london-latest.osm.pbf"
MTC_FILE = CFG / "major_trauma_centres.csv"
AE_OVERRIDE = CFG / "ae_sites.csv"

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

MAJOR_TRAUMA_CENTRES = [
    ("The Royal London Hospital", "E1 1FR"),
    ("St Mary's Hospital", "W2 1NY"),
    ("King's College Hospital", "SE5 9RS"),
    ("St George's Hospital", "SW17 0QT"),
]
# typical speeds (km/h) where OpenStreetMap has no speed limit; London is mostly 20-30 mph
TYPICAL_SPEEDS = {"motorway": 96, "motorway_link": 64, "trunk": 64, "trunk_link": 48, "primary": 48,
                  "primary_link": 40, "secondary": 40, "secondary_link": 32, "tertiary": 40,
                  "tertiary_link": 32, "unclassified": 32, "residential": 32, "living_street": 16}
BANDS = [0, 10, 15, 20, 30, np.inf]
BAND_LABELS = ["under 10 min", "10–15 min", "15–20 min", "20–30 min", "over 30 min"]
BAND_COLOURS = ["#1D5A2C", "#7BB477", "#E8C25A", "#E07B39", "#B8403A"]


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# HOSPITALS
# --------------------------------------------------------------------------
def load_trauma_centres():
    CFG.mkdir(exist_ok=True)
    if MTC_FILE.exists():
        mtc = pd.read_csv(MTC_FILE)
    else:
        mtc = pd.DataFrame(MAJOR_TRAUMA_CENTRES, columns=["name", "postcode"])
        mtc["lat"], mtc["lon"] = np.nan, np.nan
    todo = mtc["lat"].isna() | mtc["lon"].isna()
    for i in mtc.index[todo]:
        pc = mtc.at[i, "postcode"]
        try:
            r = requests.get(f"https://api.postcodes.io/postcodes/{pc.replace(' ', '')}", timeout=30)
            r.raise_for_status()
            res = r.json()["result"]
            mtc.at[i, "lat"], mtc.at[i, "lon"] = res["latitude"], res["longitude"]
        except Exception as e:
            sys.exit(f"!! Could not locate {mtc.at[i, 'name']} ({pc}): {e}\n"
                     f"   Add its latitude/longitude to {MTC_FILE} by hand and re-run.")
    mtc.to_csv(MTC_FILE, index=False)
    return gpd.GeoDataFrame(mtc, geometry=gpd.points_from_xy(mtc["lon"], mtc["lat"]), crs=WGS84).to_crs(BNG)


def ae_from_osm(osm_file):
    q = "\"emergency\"=>\"yes\""
    pts = pyogrio.read_dataframe(osm_file, layer="points", columns=["name", "other_tags"],
                                 where=f"other_tags LIKE '%\"amenity\"=>\"hospital\"%' AND other_tags LIKE '%{q}%'")
    polys = pyogrio.read_dataframe(osm_file, layer="multipolygons", columns=["name", "amenity", "other_tags"],
                                   where=f"amenity = 'hospital' AND other_tags LIKE '%{q}%'")
    polys = polys.to_crs(BNG)
    polys["geometry"] = polys.geometry.representative_point()
    ae = pd.concat([pts.to_crs(BNG)[["name", "geometry"]], polys[["name", "geometry"]]], ignore_index=True)
    return gpd.GeoDataFrame(ae, geometry="geometry", crs=BNG)


def dedupe(points, min_gap=400):
    keep = []
    for i, p in points.geometry.items():
        if all(p.distance(points.geometry[j]) >= min_gap for j in keep):
            keep.append(i)
    return points.loc[keep].reset_index(drop=True)


def load_ae_sites(osm_file, mtc):
    if AE_OVERRIDE.exists():
        log(f"  using your list in {AE_OVERRIDE.relative_to(ROOT)}")
        ae = pd.read_csv(AE_OVERRIDE)
        ae = gpd.GeoDataFrame(ae, geometry=gpd.points_from_xy(ae["lon"], ae["lat"]), crs=WGS84).to_crs(BNG)
    else:
        ae = ae_from_osm(osm_file)
    ae = pd.concat([mtc[["name", "geometry"]], ae[["name", "geometry"]]], ignore_index=True)
    ae = dedupe(gpd.GeoDataFrame(ae, geometry="geometry", crs=BNG))   # trauma centres listed first, so kept by name
    ae["name"] = ae["name"].fillna("(unnamed hospital)")
    ll = ae.to_crs(WGS84)
    RES.mkdir(exist_ok=True)
    pd.DataFrame({"name": ae["name"], "lat": ll.geometry.y.round(6), "lon": ll.geometry.x.round(6)}) \
        .to_csv(RES / "ae_sites_from_osm.csv", index=False)
    return ae


# --------------------------------------------------------------------------
# NETWORK
# --------------------------------------------------------------------------
def load_network():
    import osmnx as ox
    if not GRAPH.exists():
        sys.exit(f"Missing {GRAPH}. Run step 2 without --skip-network first.")
    G = ox.load_graphml(GRAPH)
    G = ox.truncate.largest_component(G, strongly=True)    # every node can reach every other
    G = ox.routing.add_edge_speeds(G, hwy_speeds=TYPICAL_SPEEDS, fallback=30)
    G = ox.routing.add_edge_travel_times(G)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.to_crs(BNG)
    edges = edges.to_crs(BNG)
    node_ids = np.array(nodes.index)
    pos = {n: i for i, n in enumerate(node_ids)}
    u = np.array([pos[a] for a, _, _ in edges.index])
    v = np.array([pos[b] for _, b, _ in edges.index])
    t = edges["travel_time"].astype(float).values
    n = len(node_ids)
    # REVERSED graph: running Dijkstra from the hospitals gives travel time FROM every node TO a hospital
    # parallel streets between the same two nodes: keep the quickest
    df = pd.DataFrame({"a": v, "b": u, "t": t}).groupby(["a", "b"], as_index=False)["t"].min()
    M = sparse.csr_matrix((df["t"].values, (df["a"].values, df["b"].values)), shape=(n, n))
    tree = cKDTree(np.column_stack([nodes.geometry.x, nodes.geometry.y]))
    edges_simple = edges.reset_index()[["u", "geometry"]]
    edges_simple["u_pos"] = u
    return M, tree, nodes, edges_simple


def times_to(M, tree, sites):
    """Minutes from every node to the nearest of `sites`, and which site that is."""
    _, site_nodes = tree.query(np.column_stack([sites.geometry.x, sites.geometry.y]))
    dist, _, src = dijkstra(M, directed=True, indices=site_nodes, return_predecessors=True, min_only=True)
    lookup = {nd: i for i, nd in enumerate(site_nodes)}
    nearest = np.array([lookup.get(s, -1) for s in src])
    return dist / 60.0, nearest


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-file", type=Path, default=OSM_FILE, help="OSM extract used in step 2")
    args = ap.parse_args()
    if not CONTEXT.exists():
        sys.exit(f"Missing {CONTEXT}. Run steps 1-2 first.")
    t_all = time.time()
    RES.mkdir(exist_ok=True)

    log("== 1. Hospitals ==")
    mtc = load_trauma_centres()
    log("  major trauma centres: " + ", ".join(mtc["name"]))
    ae = load_ae_sites(args.osm_file, mtc)
    log(f"  {len(ae)} A&E sites (including the trauma centres); list saved to results/ae_sites_from_osm.csv")

    log("\n== 2. Street network with travel times ==")
    t0 = time.time()
    M, tree, nodes, edges = load_network()
    log(f"  {M.shape[0]:,} junctions/nodes, {M.nnz:,} street links, ready in {time.time() - t0:.0f}s")

    log("\n== 3. Drive times ==")
    t_mtc, near_mtc = times_to(M, tree, mtc)
    t_ae, near_ae = times_to(M, tree, ae)
    cas = gpd.read_file(CONTEXT, layer="casualties").to_crs(BNG)
    gap, cas_node = tree.query(np.column_stack([cas.geometry.x, cas.geometry.y]))
    cas["snap_distance_m"] = gap.round(1)
    cas["mins_to_ae"] = t_ae[cas_node].round(1)
    cas["nearest_ae"] = ae["name"].values[near_ae[cas_node]]
    cas["mins_to_trauma_centre"] = t_mtc[cas_node].round(1)
    cas["nearest_trauma_centre"] = mtc["name"].values[near_mtc[cas_node]]
    cas["trauma_band"] = pd.cut(cas["mins_to_trauma_centre"], BANDS, labels=BAND_LABELS, right=False).astype(str)
    far = (gap > 200).sum()
    if far:
        log(f"  ! {far} casualties are more than 200 m from the drivable network (times are less reliable)")
    log(f"  done in {time.time() - t0:.0f}s")

    ksi = cas["severity"].isin(["Fatal", "Serious"])
    k = cas[ksi]

    log("\n== 4. Saving ==")
    gpkg = OUT / "london_access.gpkg"
    cas.to_file(gpkg, layer="casualties_access", driver="GPKG")
    edges["mins_to_trauma_centre"] = t_mtc[edges["u_pos"].values]
    edges["band"] = pd.cut(edges["mins_to_trauma_centre"], BANDS, labels=BAND_LABELS, right=False).astype(str)
    edges[["mins_to_trauma_centre", "band", "geometry"]].to_file(gpkg, layer="network_time_to_trauma", driver="GPKG")

    by_b = k.groupby("borough").agg(
        serious_casualties=("mins_to_trauma_centre", "size"),
        median_mins_to_trauma_centre=("mins_to_trauma_centre", "median"),
        pct_over_15_min=("mins_to_trauma_centre", lambda s: 100 * (s >= 15).mean()),
        pct_over_20_min=("mins_to_trauma_centre", lambda s: 100 * (s >= 20).mean()),
        median_mins_to_ae=("mins_to_ae", "median"),
    ).round(1).sort_values("median_mins_to_trauma_centre", ascending=False)
    by_b.to_csv(RES / "access_by_borough.csv")
    log(f"  {gpkg.name}: casualties_access, network_time_to_trauma")
    log("  results/access_by_borough.csv")

    make_figures(edges, k, mtc, ae, nodes)
    log("  figures/: access_to_trauma_care.png, access_curve.png")

    log("\n== Summary (fatal + serious pedestrian casualties) ==")
    log(f"  {len(k):,} casualties. Median drive time to the nearest A&E: {k['mins_to_ae'].median():.1f} min; "
        f"to the nearest major trauma centre: {k['mins_to_trauma_centre'].median():.1f} min")
    for cut in (10, 15, 20, 30):
        log(f"  within {cut:>2} min of a major trauma centre: {100 * (k['mins_to_trauma_centre'] < cut).mean():5.1f}%"
            f"   of an A&E: {100 * (k['mins_to_ae'] < cut).mean():5.1f}%")
    log("\nNearest major trauma centre for serious pedestrian casualties:")
    log(k["nearest_trauma_centre"].value_counts().to_string())
    log("\nBoroughs where serious casualties are furthest from a major trauma centre (median minutes):")
    log(by_b[["serious_casualties", "median_mins_to_trauma_centre", "pct_over_20_min"]].head(8).to_string())
    log("\nDrive times assume free-flowing traffic. Real journeys, especially at peak times, take longer,")
    log("and ambulances may also go to a closer trauma unit or arrive by air. Review results/ae_sites_from_osm.csv.")
    log(f"Done in {time.time() - t_all:.0f}s.")


def make_figures(edges, ksi, mtc, ae, nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    for lab, col in zip(BAND_LABELS, BAND_COLOURS):
        sub = edges[edges["band"] == lab]
        if len(sub):
            sub.plot(ax=ax, color=col, linewidth=0.35)
    ksi.plot(ax=ax, color="#111111", markersize=1.2, alpha=0.5)
    mtc.plot(ax=ax, color="white", edgecolor="#111", marker="P", markersize=180, linewidth=1.2, zorder=5)
    for _, r in mtc.iterrows():
        ax.annotate(r["name"], (r.geometry.x, r.geometry.y), xytext=(8, 8), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    handles = [Line2D([], [], color=c, lw=3, label=l) for l, c in zip(BAND_LABELS, BAND_COLOURS)]
    handles += [Line2D([], [], color="#111", marker="o", lw=0, ms=3, label="Fatal or serious pedestrian casualty"),
                Line2D([], [], color="#111", marker="P", mfc="white", lw=0, ms=10, label="Major trauma centre")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9,
              title="Drive time to the nearest\nmajor trauma centre\n(free-flowing traffic)", title_fontsize=9,
              alignment="left")
    ax.set_title("How far serious pedestrian casualties are from London's major trauma centres", loc="left", fontsize=13)
    ax.set_axis_off()
    fig.text(0.01, 0.01, "Data: DfT STATS19 (OGL); street network © OpenStreetMap contributors (ODbL)",
             fontsize=8, color="#555")
    fig.savefig(FIG / "access_to_trauma_care.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 45, 451)
    for col, lab, colour in (("mins_to_ae", "Nearest A&E", "#2F6BA8"),
                             ("mins_to_trauma_centre", "Nearest major trauma centre", "#B8403A")):
        v = np.sort(ksi[col].dropna().values)
        ax.plot(x, 100 * np.searchsorted(v, x, side="right") / max(len(v), 1), color=colour, lw=2, label=lab)
    ax.set_xlabel("Drive time, minutes (free-flowing traffic)")
    ax.set_ylabel("% of fatal + serious pedestrian casualties")
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Share of serious pedestrian casualties within X minutes of emergency care", loc="left", fontsize=12)
    fig.savefig(FIG / "access_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
