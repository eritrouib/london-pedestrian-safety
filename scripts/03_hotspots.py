"""
03_hotspots.py
London pedestrian road safety - Step 3: where are the hotspots, and how do areas compare fairly?

Three complementary views:
  1. Street hotspots: network kernel density. Streets are cut into 50 m pieces ("lixels"), each
     casualty is snapped to its street, and density is spread ALONG the network within 250 m,
     so a cluster on one road does not leak onto a parallel street across a block.
  2. Area hotspots: Getis-Ord Gi* on small areas (LSOAs), with 999 permutations, to separate
     statistically significant clusters of high (hot) and low (cold) casualty density from noise.
  3. Fair rates: the same boroughs ranked three ways (per resident, per km², per km of road),
     with Empirical Bayes smoothing so small areas with few residents don't produce extreme rates.

Usage (from the project folder, after steps 1 and 2 with the street network):
    python scripts/03_hotspots.py

Outputs:
    data/processed/london_hotspots.gpkg   layers: street_hotspots, area_hotspots, borough_rates
    figures/street_hotspots.png           serious-injury density along London's streets
    figures/area_hotspots.png             Gi* hot and cold spots
    figures/borough_rank_change.png       how borough rankings change with the measure used
    figures/hotspots_map.html             interactive map of the above
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
CONTEXT = OUT / "london_context.gpkg"
GRAPH = RAW / "osm" / "london_drive.graphml"

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"
LIXEL_M = 50          # length of street pieces
BANDWIDTH_M = 250     # how far along the street each casualty's influence spreads
SNAP_M = 30           # casualties further than this from a drivable street are not snapped
PERMUTATIONS = 999
SEED = 42

GI_COLOURS = {"Hot spot 99%": "#B8403A", "Hot spot 95%": "#E07B39", "Hot spot 90%": "#F2B880",
              "Not significant": "#EEEEEE",
              "Cold spot 90%": "#B9D3E8", "Cold spot 95%": "#6FA5D6", "Cold spot 99%": "#2F6BA8"}


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. STREET HOTSPOTS: network kernel density on lixels
# --------------------------------------------------------------------------
def load_edges():
    try:
        import osmnx as ox
    except ImportError:
        sys.exit("osmnx is needed: python -m pip install osmnx")
    if not GRAPH.exists():
        sys.exit(f"Missing {GRAPH}. Run step 2 without --skip-network first.")
    G = ox.convert.to_undirected(ox.load_graphml(GRAPH))   # one edge per street, not one per direction
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(BNG).reset_index()
    for c in ("name", "highway"):
        if c not in edges.columns:
            edges[c] = None
        edges[c] = edges[c].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v)
    return edges[["name", "highway", "geometry"]]


def lixelize(edges, step):
    """Cut every street into pieces of about `step` metres."""
    lengths = edges.length.values
    n = np.maximum(1, np.ceil(lengths / step).astype(int))
    idx = np.repeat(np.arange(len(edges)), n)
    k = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)   # position of each piece within its street
    f0, f1 = k / n[idx], (k + 1) / n[idx]
    geoms = edges.geometry.values[idx]
    c0 = shapely.get_coordinates(shapely.line_interpolate_point(geoms, f0, normalized=True))
    c1 = shapely.get_coordinates(shapely.line_interpolate_point(geoms, f1, normalized=True))
    lines = shapely.linestrings(np.stack([c0, c1], axis=1))
    lix = gpd.GeoDataFrame({"name": edges["name"].values[idx], "highway": edges["highway"].values[idx]},
                           geometry=lines, crs=BNG)
    return lix, np.round(c0, 1), np.round(c1, 1)


def adjacency(c0, c1):
    """Pieces that share an end point are neighbours."""
    keys, inverse = np.unique(np.vstack([c0, c1]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(c0)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    B = sparse.csr_matrix((np.ones(2 * n, dtype=np.float32), (rows, inverse)), shape=(n, len(keys)))
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    return A


def network_kde(A, counts, step, bandwidth):
    """Quartic kernel along the network, using hop distance between pieces."""
    hops = int(bandwidth // step)
    weight = lambda d: (1 - (d / bandwidth) ** 2) ** 2 if d < bandwidth else 0.0
    n = A.shape[0]
    one_step = (sparse.identity(n, format="csr", dtype=np.float32) + A).tocsr()
    reach = sparse.identity(n, format="csr", dtype=np.float32)
    density = {k: weight(0) * v for k, v in counts.items()}
    for h in range(1, hops + 1):
        w = weight(h * step)
        if w <= 0:
            break
        new = reach @ one_step
        new.data[:] = 1
        exact = (new - reach).tocsr()          # pieces exactly h steps away
        exact.eliminate_zeros()
        for k, v in counts.items():
            density[k] = density[k] + w * (exact @ v)
        reach = new
    return density


def pick_top_locations(lix, column, n=10, min_gap_m=750):
    """Highest-density pieces, at least `min_gap_m` apart so one road isn't listed ten times."""
    cand = lix[lix[column] > 0].sort_values(column, ascending=False)
    chosen = []
    cents = cand.geometry.centroid
    for i in cand.index:
        p = cents.loc[i]
        if all(p.distance(cents.loc[j]) >= min_gap_m for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return lix.loc[chosen]


# --------------------------------------------------------------------------
# 2. AREA HOTSPOTS: Getis-Ord Gi*
# --------------------------------------------------------------------------
def gi_star(lsoa, column):
    from libpysal.weights import Queen
    from esda.getisord import G_Local
    import esda
    w = Queen.from_dataframe(lsoa, use_index=False, silence_warnings=True)
    g = G_Local(lsoa[column].values.astype(float), w, transform="B", star=True,
                permutations=PERMUTATIONS, seed=SEED, keep_simulations=False)
    z, p = g.Zs, g.p_sim
    cls = np.full(len(z), "Not significant", dtype=object)
    for conf, cut in ((90, 0.10), (95, 0.05), (99, 0.01)):
        cls[(p <= cut) & (z > 0)] = f"Hot spot {conf}%"
        cls[(p <= cut) & (z < 0)] = f"Cold spot {conf}%"
    fdr_cut = esda.fdr(p, 0.05)
    return z, p, cls, p <= fdr_cut


# --------------------------------------------------------------------------
# 3. FAIR RATES
# --------------------------------------------------------------------------
def empirical_bayes(events, population):
    """Global Empirical Bayes smoothing (Marshall 1991): pulls unstable small-area rates towards the mean."""
    events, population = np.asarray(events, float), np.asarray(population, float)
    ok = population > 0
    b = events[ok].sum() / population[ok].sum()
    r = np.where(ok, events / np.where(ok, population, 1), np.nan)
    s2 = np.nansum(population * (r - b) ** 2) / population[ok].sum()
    a = max(s2 - b / population[ok].mean(), 0)
    w = np.where(ok, a / (a + b / np.where(ok, population, 1)), 0)
    return np.where(ok, w * r + (1 - w) * b, np.nan)


# --------------------------------------------------------------------------
# FIGURES
# --------------------------------------------------------------------------
def make_figures(lix, lsoa, boroughs, years, edges=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    FIG.mkdir(exist_ok=True)
    outline = boroughs.boundary

    # street hotspots
    fig, ax = plt.subplots(figsize=(11, 9))
    if edges is not None:
        edges.plot(ax=ax, color="#D9DEDB", linewidth=0.25)   # the whole street network, faintly
    outline.plot(ax=ax, color="#9AA5A0", linewidth=0.5)
    pos = lix[lix["ksi_density"] > 0].sort_values("ksi_density")
    if len(pos):
        vmax = pos["ksi_density"].quantile(0.995)
        pos.plot(ax=ax, column="ksi_density", cmap="inferno_r", linewidth=0.9, vmin=0, vmax=vmax,
                 legend=True, legend_kwds={"label": "Fatal + serious pedestrian casualties\n"
                                                    "(kernel-weighted, within 250 m along streets, per year)",
                                           "shrink": 0.6})
    ax.set_title(f"Where serious pedestrian injuries concentrate on London's streets, {years}",
                 loc="left", fontsize=13)
    ax.set_axis_off()
    fig.text(0.01, 0.01, "Data: DfT STATS19 (OGL); street network © OpenStreetMap contributors (ODbL)",
             fontsize=8, color="#555")
    fig.savefig(FIG / "street_hotspots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # area hotspots
    fig, ax = plt.subplots(figsize=(11, 9))
    lsoa.plot(ax=ax, color=lsoa["gi_class"].map(GI_COLOURS), linewidth=0)
    outline.plot(ax=ax, color="#555", linewidth=0.5)
    handles = [Patch(color=c, label=l) for l, c in GI_COLOURS.items() if (lsoa["gi_class"] == l).any()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"Hot and cold spots of pedestrian casualties (Getis-Ord Gi*), {years}", loc="left", fontsize=13)
    ax.set_axis_off()
    fig.text(0.01, 0.01, "Small areas (LSOA 2021) by casualties per km² per year; 999 permutations. "
             "Data: DfT STATS19, ONS (OGL)", fontsize=8, color="#555")
    fig.savefig(FIG / "area_hotspots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # rank change (slope chart)
    b = boroughs.sort_values("rank_per_resident")
    fig, ax = plt.subplots(figsize=(8, 10))
    for _, r in b.iterrows():
        moved = abs(r["rank_per_resident"] - r["rank_per_road_km"]) >= 8
        col = "#B8403A" if moved else "#9AA5A0"
        ax.plot([0, 1], [r["rank_per_resident"], r["rank_per_road_km"]], color=col, lw=1.6 if moved else 0.9)
        ax.text(-0.03, r["rank_per_resident"], r["borough"], ha="right", va="center", fontsize=8,
                color="#222" if moved else "#666")
        ax.text(1.03, r["rank_per_road_km"], r["borough"], ha="left", va="center", fontsize=8,
                color="#222" if moved else "#666")
    ax.set_ylim(len(b) + 1, 0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["Per resident", "Per km of road"])
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same data, different ranking:\nfatal + serious pedestrian casualties by borough (1 = highest)",
                 loc="left", fontsize=12)
    fig.text(0.01, 0.01, "Red: boroughs that move 8 or more places. Data: DfT STATS19, MHCLG IoD 2025 population, "
             "OpenStreetMap road lengths", fontsize=8, color="#555")
    fig.savefig(FIG / "borough_rank_change.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # interactive map
    try:
        import folium
        m = folium.Map(location=[51.505, -0.11], zoom_start=11, tiles=None)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/"
                         "MapServer/tile/{z}/{y}/{x}", attr="Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors",
                         name="Light grey", max_zoom=16).add_to(m)
        sig = lsoa[lsoa["gi_class"] != "Not significant"].to_crs(WGS84)
        sig = sig.astype({"casualties_per_km2_yr": float})
        folium.GeoJson(sig[["lsoa21nm", "gi_class", "casualties_per_km2_yr", "geometry"]], name="Area hot and cold spots",
                       style_function=lambda f: {"fillColor": GI_COLOURS[f["properties"]["gi_class"]], "color": "none",
                                                 "fillOpacity": 0.6},
                       tooltip=folium.GeoJsonTooltip(["lsoa21nm", "gi_class", "casualties_per_km2_yr"],
                                                     ["Area", "Result", "Casualties per km² per year"])).add_to(m)
        top = lix[lix["ksi_density"] > 0]
        top = top[top["ksi_density"] >= top["ksi_density"].quantile(0.95)].to_crs(WGS84)
        top = top.astype({"ksi_density": float})
        vmax = float(top["ksi_density"].max()) or 1.0
        folium.GeoJson(top[["name", "ksi_density", "geometry"]], name="Street hotspots (top 5%)",
                       style_function=lambda f: {"color": "#7A0C2E", "weight": 2 + 4 * f["properties"]["ksi_density"] / vmax},
                       tooltip=folium.GeoJsonTooltip(["name", "ksi_density"], ["Street", "Density"])).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(FIG / "hotspots_map.html")
        log("  figures/hotspots_map.html")
    except Exception as ex:
        log(f"  (interactive map skipped: {ex})")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not CONTEXT.exists():
        sys.exit(f"Missing {CONTEXT}. Run step 2 first.")
    t_all = time.time()

    cas = gpd.read_file(CONTEXT, layer="casualties").to_crs(BNG)
    lsoa = gpd.read_file(CONTEXT, layer="lsoa").to_crs(BNG)
    boroughs = gpd.read_file(CONTEXT, layer="boroughs").to_crs(BNG)
    years = int(cas["year"].nunique()) if "year" in cas.columns else 5
    yr_label = f"{int(cas['year'].min())}–{int(cas['year'].max())}" if "year" in cas.columns else ""
    ksi = cas["severity"].isin(["Fatal", "Serious"]).values

    log("== 1. Street hotspots (network kernel density) ==")
    t0 = time.time()
    edges = load_edges()
    lix, c0, c1 = lixelize(edges, LIXEL_M)
    log(f"  {len(edges):,} streets cut into {len(lix):,} pieces of about {LIXEL_M} m")
    A = adjacency(c0, c1)
    snap = gpd.sjoin_nearest(cas[["geometry"]], lix[["geometry"]], how="left", max_distance=SNAP_M)
    snap = snap[~snap.index.duplicated()]
    on_net = snap["index_right"].notna().values
    log(f"  {on_net.sum():,} of {len(cas):,} casualties within {SNAP_M} m of a drivable street")
    idx = snap["index_right"].values
    counts = {
        "all": np.bincount(idx[on_net].astype(int), minlength=len(lix)).astype(np.float32),
        "ksi": np.bincount(idx[on_net & ksi].astype(int), minlength=len(lix)).astype(np.float32),
    }
    dens = network_kde(A, counts, LIXEL_M, BANDWIDTH_M)
    lix["casualties"] = counts["all"].astype(int)
    lix["ksi"] = counts["ksi"].astype(int)
    lix["all_density"] = (dens["all"] / years).round(4)
    lix["ksi_density"] = (dens["ksi"] / years).round(4)
    log(f"  density calculated in {time.time() - t0:.0f}s")

    top = pick_top_locations(lix, "ksi_density", n=10)
    top_b = gpd.sjoin(gpd.GeoDataFrame(top.drop(columns="geometry"), geometry=top.geometry.centroid, crs=BNG),
                      boroughs[["borough", "geometry"]], predicate="within", how="left")

    log("\n== 2. Area hotspots (Getis-Ord Gi*) ==")
    z, p, cls, fdr = gi_star(lsoa, "casualties_per_km2_yr")
    lsoa["gi_z"], lsoa["gi_p"], lsoa["gi_class"], lsoa["gi_significant_fdr"] = z.round(3), p, cls, fdr
    counts_cls = pd.Series(cls).value_counts()
    log("  " + ", ".join(f"{k}: {v:,}" for k, v in counts_cls.items()))
    log(f"  {int(fdr.sum()):,} areas remain significant after correcting for multiple testing (FDR 5%)")

    log("\n== 3. Fair rates ==")
    lsoa["road_km_total"] = lsoa[[c for c in lsoa.columns if c.startswith("road_km_")]].sum(axis=1)
    lsoa["ksi_per_road_km_yr"] = np.where(lsoa["road_km_total"] > 0,
                                          lsoa["n_ksi"] / lsoa["road_km_total"] / years, np.nan)
    lsoa["ksi_per_1000_pop_yr_eb"] = 1000 * empirical_bayes(lsoa["n_ksi"], lsoa["population"] * years)
    boroughs["road_km_total"] = boroughs[[c for c in boroughs.columns if c.startswith("road_km_")]].sum(axis=1)
    boroughs["ksi_per_road_km_yr"] = boroughs["n_ksi"] / boroughs["road_km_total"] / years
    boroughs["rank_per_resident"] = boroughs["ksi_per_100k_pop_yr"].rank(ascending=False, method="min").astype(int)
    boroughs["rank_per_km2"] = boroughs["ksi_per_km2_yr"].rank(ascending=False, method="min").astype(int)
    boroughs["rank_per_road_km"] = boroughs["ksi_per_road_km_yr"].rank(ascending=False, method="min").astype(int)

    log("\n== 4. Saving ==")
    gpkg = OUT / "london_hotspots.gpkg"
    lix[lix["all_density"] > 0].to_file(gpkg, layer="street_hotspots", driver="GPKG")
    lsoa.to_file(gpkg, layer="area_hotspots", driver="GPKG")
    boroughs.to_file(gpkg, layer="borough_rates", driver="GPKG")
    log(f"  {gpkg.name}: street_hotspots, area_hotspots, borough_rates")
    make_figures(lix, lsoa, boroughs, yr_label, edges)   # also writes the interactive map
    log("  figures/: street_hotspots.png, area_hotspots.png, borough_rank_change.png")

    log("\n== Summary ==")
    log("Ten street locations where serious pedestrian injuries concentrate most (at least 750 m apart):")
    for i, (_, r) in enumerate(top_b.iterrows(), 1):
        name = r["name"] if isinstance(r["name"], str) and r["name"] else "(unnamed street)"
        log(f"  {i:>2}. {name:<38} {str(r.get('borough') or ''):<24} density {r['ksi_density']:.2f}")
    log("\nBorough rankings by fatal + serious casualties, three ways (1 = highest):")
    t = boroughs.set_index("borough")[["rank_per_resident", "rank_per_km2", "rank_per_road_km",
                                       "ksi_per_road_km_yr"]].sort_values("rank_per_road_km").head(12)
    t.columns = ["per resident", "per km²", "per km road", "KSI per km road/yr"]
    log(t.round(3).to_string())
    log(f"\nDone in {time.time() - t_all:.0f}s. Open figures/hotspots_map.html to explore.")


if __name__ == "__main__":
    main()
