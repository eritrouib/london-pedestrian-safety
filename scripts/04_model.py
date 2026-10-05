"""
04_model.py
London pedestrian road safety - Step 4: what about the built environment goes with pedestrian injuries?

Models pedestrian casualty counts per small area (LSOA) with negative binomial regression, the
standard model for collision counts (they are over-dispersed: a few areas have very many).

  Model A  activity only     how busy an area is: population, stations, bus stops, pubs/bars
  Model B  + street design   major roads, junctions, crossings, signals, schools
           + deprivation     the IoD income score (not the full IMD, which includes a road injury indicator)

Comparing A and B shows which street-design features still matter once busyness is accounted for.
Both are fitted for all pedestrian casualties and for fatal + serious ones (KSI), with road length
as the exposure, so results are per km of road.

Also:
  - collinearity check (VIF); variables that mostly duplicate others are dropped and reported
  - Moran's I on the residuals: is there spatial pattern the model misses?
  - inner vs outer London: do the relationships differ?
  - "more casualties than expected": small areas with significantly more casualties than the model
    predicts for a place like them, i.e. candidates for a closer look
  - optional geographically weighted regression (--gwr): local effects across London (slow)

Usage (from the project folder, after steps 1-3):
    python scripts/04_model.py
    python scripts/04_model.py --gwr     # adds geographically weighted Poisson regression (20-60 min)

Outputs:
    results/model_coefficients.csv     all effects with confidence intervals
    results/model_summary.txt          full statistical output
    results/excess_areas.csv           small areas with more casualties than expected
    data/processed/london_model.gpkg   small areas with expected counts and residuals
    figures/model_effects.png          effect sizes, all vs serious casualties
    figures/inner_outer_effects.png    effects in inner vs outer London
    figures/excess_casualties.png      observed vs expected map
    figures/gwr_local_effects.png      (with --gwr) local effects
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
CONTEXT = OUT / "london_context.gpkg"

# Inner London as defined in the London Government Act 1963
INNER_LONDON = {"City of London", "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham", "Islington",
                "Kensington and Chelsea", "Lambeth", "Lewisham", "Southwark", "Tower Hamlets", "Wandsworth",
                "Westminster"}

# name in data -> (label for people, transform, group)
VARIABLES = {
    "pop_density":             ("Population density",            "log", "activity"),
    "station_per_km2":         ("Rail/Tube stations per km²",    "log", "activity"),
    "bus_stop_per_km2":        ("Bus stops per km²",             "log", "activity"),
    "pub_bar_per_km2":         ("Pubs and bars per km²",         "log", "activity"),
    "major_road_share":        ("Share of roads that are major", "raw", "street"),
    "junction_density":        ("Junctions per km²",             "log", "street"),
    "crossing_per_km2":        ("Crossings per km²",             "log", "street"),
    "traffic_signals_per_km2": ("Traffic signals per km²",       "log", "street"),
    "school_per_km2":          ("Schools per km²",               "log", "street"),
    "income_score":            ("Income deprivation",            "raw", "deprivation"),
}
OUTCOMES = {"n_casualties": "All pedestrian casualties", "n_ksi": "Fatal + serious"}
VIF_LIMIT = 10


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def prepare(lsoa):
    df = lsoa.copy()
    df["road_km_total"] = df[[c for c in df.columns if c.startswith("road_km_")]].sum(axis=1)
    available = [v for v in VARIABLES if v in df.columns]
    missing = [v for v in VARIABLES if v not in df.columns]
    if missing:
        log(f"  ! not in the data, skipped: {missing}")
    keep = (df["road_km_total"] > 0.05)
    for v in available:
        keep &= df[v].notna()
    dropped = (~keep).sum()
    df = df[keep].copy()
    if dropped:
        log(f"  {dropped} small areas without roads or with missing values left out")
    Z = pd.DataFrame(index=df.index)
    for v in available:
        x = df[v].astype(float)
        x = np.log1p(x.clip(lower=0)) if VARIABLES[v][1] == "log" else x
        sd = x.std()
        Z[v] = (x - x.mean()) / sd if sd > 0 else 0.0
    Z = Z.loc[:, Z.std() > 0]
    df["inner"] = df["borough"].isin(INNER_LONDON)
    return df, Z


def vif_filter(Z):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    cols = list(Z.columns)
    dropped = []
    while True:
        X = sm.add_constant(Z[cols])
        vifs = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=cols)
        if vifs.max() <= VIF_LIMIT or len(cols) <= 2:
            return cols, vifs, dropped
        worst = vifs.idxmax()
        dropped.append((worst, vifs.max()))
        cols.remove(worst)


# --------------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------------
def fit_nb(y, Z, cols, exposure):
    import statsmodels.api as sm
    X = sm.add_constant(Z[cols], has_constant="add")
    nb = sm.NegativeBinomial(y, X, exposure=exposure).fit(disp=0, maxiter=500, cov_type="HC1")
    pois = sm.Poisson(y, X, exposure=exposure).fit(disp=0, maxiter=200)
    return nb, pois, X


def effects_table(res, cols, outcome, model, subset="All London"):
    ci = res.conf_int()
    rows = []
    for c in cols:
        b, lo, hi = res.params[c], ci.loc[c, 0], ci.loc[c, 1]
        rows.append(dict(outcome=outcome, model=model, subset=subset, variable=c, label=VARIABLES[c][0],
                         group=VARIABLES[c][2], coef=b, pct_change=100 * (np.exp(b) - 1),
                         pct_low=100 * (np.exp(lo) - 1), pct_high=100 * (np.exp(hi) - 1),
                         p_value=res.pvalues[c]))
    return pd.DataFrame(rows)


def morans_i(gdf, values):
    from libpysal.weights import Queen
    from esda.moran import Moran
    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "R"
    m = Moran(np.asarray(values, float), w, permutations=999)
    return m.I, m.p_sim


def excess_areas(df, y, mu, alpha):
    from scipy.stats import nbinom
    n = 1.0 / alpha
    p = n / (n + mu)
    p_excess = nbinom.sf(y - 1, n, p)          # P(at least this many | expected mu)
    out = df[["lsoa21cd", "lsoa21nm", "borough"]].copy()
    out["observed"], out["expected"] = y, np.round(mu, 1)
    out["ratio"] = np.round((y + 1) / (mu + 1), 2)
    out["p_excess"] = p_excess
    # Benjamini-Hochberg: with thousands of areas some look extreme by chance alone
    order = np.argsort(p_excess)
    ranked = p_excess[order] * len(p_excess) / np.arange(1, len(p_excess) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out["q_value"] = np.empty_like(q)
    out.iloc[order, out.columns.get_loc("q_value")] = np.minimum(q, 1)
    out["significant_fdr5"] = out["q_value"] < 0.05
    return out


# --------------------------------------------------------------------------
# GWR (optional)
# --------------------------------------------------------------------------
def run_gwr(df, Z, cols, y, exposure):
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    from spglm.family import Poisson
    cents = df.geometry.centroid
    coords = np.column_stack([cents.x, cents.y])
    yy = np.asarray(y, float).reshape(-1, 1)
    X = Z[cols].values
    off = np.asarray(exposure, float).reshape(-1, 1)
    log("  choosing the bandwidth (number of neighbours) ...")
    bw = Sel_BW(coords, yy, X, family=Poisson(), offset=off).search()
    log(f"  bandwidth: {int(bw)} nearest small areas")
    res = GWR(coords, yy, X, bw, family=Poisson(), offset=off).fit()
    sig = res.filter_tvals()          # zero where not significant after multiple-testing correction
    local = pd.DataFrame(res.params[:, 1:], columns=cols, index=df.index)
    local_sig = pd.DataFrame(sig[:, 1:], columns=cols, index=df.index)
    return bw, local, local_sig


# --------------------------------------------------------------------------
# FIGURES
# --------------------------------------------------------------------------
def forest(ax, tab, title, colours):
    groups_order = ["activity", "street", "deprivation"]
    labels = [l for g in groups_order for l in tab[tab["group"] == g]["label"].unique()]
    ys = {l: i for i, l in enumerate(reversed(labels))}
    keys = list(colours)
    for k, (key, colour) in enumerate(colours.items()):
        sub = tab[tab["key"] == key]
        off = (k - (len(keys) - 1) / 2) * 0.22
        for _, r in sub.iterrows():
            yv = ys[r["label"]] + off
            ax.plot([r["pct_low"], r["pct_high"]], [yv, yv], color=colour, lw=1.6)
            ax.plot(r["pct_change"], yv, "o", color=colour, ms=5,
                    mfc=colour if r["p_value"] < 0.05 else "white")
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_yticks(list(ys.values()), list(ys.keys()), fontsize=9)
    ax.set_xlabel("% change in casualties per km of road, for an area one standard deviation higher", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", lw=1.6, label=k) for k, c in colours.items()]
    handles.append(Line2D([], [], color="#555", marker="o", mfc="white", lw=0, label="not significant (p ≥ 0.05)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")


def make_figures(coefs, gdf, local=None, local_sig=None, cols=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)

    t = coefs[(coefs["model"] == "B: activity + street + deprivation") & (coefs["subset"] == "All London")].copy()
    t["key"] = t["outcome"]
    fig, ax = plt.subplots(figsize=(9, 6))
    forest(ax, t, "What goes with more pedestrian casualties, once busyness is accounted for",
           {"All pedestrian casualties": "#2F6BA8", "Fatal + serious": "#B8403A"})
    fig.text(0.01, -0.02, "Negative binomial regression on London small areas (LSOA 2021), exposure = km of road. "
             "Bars: 95% confidence intervals (robust).", fontsize=8, color="#555")
    fig.savefig(FIG / "model_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    t = coefs[(coefs["model"] == "B: activity + street + deprivation") & (coefs["outcome"] == "All pedestrian casualties")
              & (coefs["subset"] != "All London")].copy()
    if len(t):
        t["key"] = t["subset"]
        fig, ax = plt.subplots(figsize=(9, 6))
        forest(ax, t, "Do the relationships differ between inner and outer London?",
               {"Inner London": "#7A3E9D", "Outer London": "#3E8E4E"})
        fig.savefig(FIG / "inner_outer_effects.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    bins = [0, 0.5, 0.8, 1.25, 2, np.inf]
    labels = ["Much fewer than expected (<0.5×)", "Fewer (0.5–0.8×)", "About as expected",
              "More (1.25–2×)", "Much more than expected (>2×)"]
    cols_ = ["#2F6BA8", "#9CC3E4", "#EEEEEE", "#F2B880", "#B8403A"]
    gdf = gdf.copy()
    gdf["cat"] = pd.cut(gdf["ratio"], bins=bins, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    gdf.plot(ax=ax, color=gdf["cat"].map(dict(zip(labels, cols_))).astype(object).fillna("#EEEEEE"), linewidth=0)
    sig = gdf[gdf["significant_fdr5"] & (gdf["observed"] > gdf["expected"])]
    if len(sig):
        sig.boundary.plot(ax=ax, color="#111", linewidth=0.6)
    gdf.dissolve("borough").boundary.plot(ax=ax, color="#666", linewidth=0.4)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in zip(labels, cols_)]
    handles.append(Patch(facecolor="none", edgecolor="#111",
                         label="Significantly more than expected (after multiple-testing correction)"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Pedestrian casualties compared with what the model expects for each small area",
                 loc="left", fontsize=13)
    ax.set_axis_off()
    fig.savefig(FIG / "excess_casualties.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if local is not None:
        show = [c for c in ("major_road_share", "pub_bar_per_km2", "junction_density", "income_score") if c in cols][:4]
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        for ax, c in zip(axes.ravel(), show):
            g = gdf.copy()
            g["v"] = local[c].values
            g.loc[local_sig[c].values == 0, "v"] = np.nan
            lim = np.nanmax(np.abs(g["v"])) if np.isfinite(np.nanmax(np.abs(g["v"]))) else 1
            g.plot(ax=ax, column="v", cmap="RdBu_r", vmin=-lim, vmax=lim, linewidth=0, legend=True,
                   missing_kwds={"color": "#EEEEEE"}, legend_kwds={"shrink": 0.6})
            ax.set_title(VARIABLES[c][0], loc="left", fontsize=11)
            ax.set_axis_off()
        for ax in axes.ravel()[len(show):]:
            ax.set_axis_off()
        fig.suptitle("Local effects (geographically weighted Poisson regression); grey = not significant",
                     x=0.01, ha="left", fontsize=13)
        fig.savefig(FIG / "gwr_local_effects.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gwr", action="store_true", help="also run geographically weighted Poisson regression (slow)")
    args = ap.parse_args()
    if not CONTEXT.exists():
        sys.exit(f"Missing {CONTEXT}. Run steps 1-3 first.")
    RES.mkdir(exist_ok=True)
    t_all = time.time()

    log("== 1. Data ==")
    lsoa = gpd.read_file(CONTEXT, layer="lsoa")
    df, Z = prepare(lsoa)
    log(f"  {len(df):,} small areas, {Z.shape[1]} explanatory variables")

    log("\n== 2. Overlap between variables (VIF) ==")
    cols, vifs, dropped = vif_filter(Z)
    for v, x in dropped:
        log(f"  dropped {VARIABLES[v][0]} (VIF {x:.1f}: mostly duplicates the others)")
    log("  kept: " + ", ".join(f"{VARIABLES[c][0]} ({vifs[c]:.1f})" for c in cols))
    activity = [c for c in cols if VARIABLES[c][2] == "activity"]

    exposure = df["road_km_total"].values
    summaries, tables = [], []
    fitted = {}
    log("\n== 3. Models ==")
    for ycol, ylabel in OUTCOMES.items():
        y = df[ycol].astype(int).values
        for mname, mcols in (("A: activity only", activity), ("B: activity + street + deprivation", cols)):
            nb, pois, X = fit_nb(y, Z, mcols, exposure)
            tables.append(effects_table(nb, mcols, ylabel, mname))
            summaries.append(f"\n{'=' * 90}\n{ylabel} | {mname}\n{'=' * 90}\n{nb.summary()}\n")
            fitted[(ycol, mname)] = (nb, pois, X)
            alpha = nb.params["alpha"]
            log(f"  {ylabel:<26} {mname:<36} AIC {nb.aic:9.1f}  pseudo-R² {nb.prsquared:.3f}  "
                f"(Poisson AIC {pois.aic:.0f}; overdispersion α = {alpha:.2f})")

        # inner vs outer London, full model
        for name, mask in (("Inner London", df["inner"].values), ("Outer London", ~df["inner"].values)):
            if mask.sum() > len(cols) * 15:
                nb, _, _ = fit_nb(y[mask], Z[mask], cols, exposure[mask])
                tables.append(effects_table(nb, cols, ylabel, "B: activity + street + deprivation", name))

    coefs = pd.concat(tables, ignore_index=True)
    coefs.round(4).to_csv(RES / "model_coefficients.csv", index=False)
    (RES / "model_summary.txt").write_text("".join(summaries), encoding="utf-8")

    log("\n== 4. Checks and expected casualties ==")
    nb, pois, X = fitted[("n_casualties", "B: activity + street + deprivation")]
    y = df["n_casualties"].astype(int).values
    mu = nb.predict(X, exposure=exposure)
    alpha = nb.params["alpha"]
    pearson = (y - mu) / np.sqrt(mu + alpha * mu ** 2)
    gdf = gpd.GeoDataFrame(df[["lsoa21cd", "lsoa21nm", "borough", "geometry"]].copy(), geometry="geometry")
    I, p_I = morans_i(gdf, pearson)
    log(f"  Moran's I of residuals: {I:.3f} (p = {p_I:.3f})" +
        ("  -> spatial pattern remains; see --gwr" if p_I < 0.05 and I > 0.05 else ""))
    ex = excess_areas(df, y, mu, alpha)
    gdf = gdf.join(ex[["observed", "expected", "ratio", "p_excess", "q_value", "significant_fdr5"]])
    gdf["pearson_residual"] = np.round(pearson, 3)
    top_ex = ex[(ex["observed"] > ex["expected"])].sort_values("p_excess").head(15)
    n_sig = int((ex["significant_fdr5"] & (ex["observed"] > ex["expected"])).sum())
    log(f"  {n_sig} small areas have significantly more casualties than expected after correcting for "
        f"multiple testing (FDR 5%)")
    top_ex.to_csv(RES / "excess_areas.csv", index=False)
    gdf.to_file(OUT / "london_model.gpkg", layer="lsoa_model", driver="GPKG")

    local = local_sig = None
    if args.gwr:
        log("\n== 5. Geographically weighted Poisson regression ==")
        t0 = time.time()
        try:
            bw, local, local_sig = run_gwr(df, Z, cols, y, exposure)
            for c in cols:
                gdf[f"gwr_{c}"] = local[c].values
            gdf.to_file(OUT / "london_model.gpkg", layer="lsoa_model", driver="GPKG")
            share = {VARIABLES[c][0]: (local_sig[c] != 0).mean() for c in cols}
            log(f"  done in {(time.time() - t0) / 60:.0f} min. Share of areas where each effect is significant:")
            for k, v in sorted(share.items(), key=lambda x: -x[1]):
                log(f"    {k:<32} {v:.0%}")
            log("  Note: Poisson GWR ignores over-dispersion, so it overstates significance. Treat the local")
            log("  maps as exploratory: where effects may be stronger or weaker, not firm local estimates.")
        except Exception as e:
            log(f"  ! GWR failed: {e}")

    log("\n== 6. Saving ==")
    make_figures(coefs, gdf, local, local_sig, cols)
    log("  results/: model_coefficients.csv, model_summary.txt, excess_areas.csv")
    log("  figures/: model_effects.png, inner_outer_effects.png, excess_casualties.png"
        + (", gwr_local_effects.png" if local is not None else ""))

    log("\n== Summary ==")
    full = coefs[(coefs["model"].str.startswith("B")) & (coefs["subset"] == "All London")]
    for ylabel in OUTCOMES.values():
        log(f"\n{ylabel}: % change per 1 SD, controlling for everything else (* = p < 0.05)")
        t = full[full["outcome"] == ylabel].sort_values("pct_change", key=abs, ascending=False)
        for _, r in t.iterrows():
            star = "*" if r["p_value"] < 0.05 else " "
            log(f"  {r['label']:<32} {r['pct_change']:+6.1f}%  [{r['pct_low']:+.1f}, {r['pct_high']:+.1f}] {star}")
    log("\nSmall areas with the most casualties beyond what the model expects (✓ = survives FDR correction):")
    for _, r in top_ex.head(10).iterrows():
        tick = "✓" if r["significant_fdr5"] else " "
        log(f"  {tick} {r['lsoa21nm']:<32} observed {r['observed']:>3}  expected {r['expected']:>5}  "
            f"p = {r['p_excess']:.1e}")
    log("\nThese are associations across areas, not proof of cause. See the README for interpretation.")
    log(f"Done in {time.time() - t_all:.0f}s.")


if __name__ == "__main__":
    main()
