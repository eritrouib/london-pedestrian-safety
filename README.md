# London Pedestrian Safety

Where do pedestrian collisions concentrate on London's roads, how does the built environment relate to them, and how far are those places from emergency care?

A spatial analysis and interactive dashboard built entirely on open data and open-source tools, with a plain-English AI question-answering layer planned on top. It extends my thesis on pedestrian collisions and the built environment in Athens to London.

**[Open the interactive dashboard](https://eritrouib.github.io/london-pedestrian-safety/)**

## Status

| Step | What | Status |
|---|---|---|
| 1 | Pedestrian casualties from DfT STATS19, cleaned and mapped | Done |
| 2 | Boundaries, population, deprivation, street network and built environment | Done |
| 3 | Hotspot analysis (network kernel density, Getis-Ord Gi*) and fair rates | Done |
| 4 | Built environment model (negative binomial, geographically weighted) | Done |
| 5 | Access to care: network travel time to A&E and major trauma centres | Done |
| 6 | Interactive dashboard | Done |
| 7 | AI assistant: plain-English questions answered by tested queries on the data | Done (switch on with `ai/SETUP.md`) |

## Getting started

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/01_get_stats19.py
```

The first run downloads about 150 MB from the Department for Transport into `data/raw/` and writes results to `data/processed/`:

| File | Contents |
|---|---|
| `london_pedestrian_safety.gpkg` | `casualties` layer (one row per pedestrian casualty) and `collisions` layer (one row per collision injuring a pedestrian), British National Grid |
| `london_pedestrian_casualties.parquet` | Casualties as GeoParquet, for fast analysis |
| `london_quicklook_map.html` | Fatal and serious casualties on an interactive map |
| `columns_report.csv` | Every source column, how complete it is and example values |

Options: `--area england` or `--area gb` for a wider area, `--refresh` to re-download, `--no-download` to reuse files already downloaded.

### Step 2: context

```bash
python scripts/02_build_context.py                 # full run, including the street network
python scripts/02_build_context.py --skip-network  # quicker first run, no junction density
```

Downloads deprivation and population figures, small-area boundaries and the Greater London OpenStreetMap extract (about 120 MB), plus the drivable street network on a full run (10–20 minutes the first time, cached afterwards). Writes:

| File | Contents |
|---|---|
| `london_context.gpkg` → `lsoa` | All London small areas (LSOA 2021, about 5,000) with population, deprivation, street features, road density, junction density and casualty counts and rates |
| `london_context.gpkg` → `boroughs` | The same, summed up to the 33 boroughs |
| `london_context.gpkg` → `casualties` | Step 1 casualties plus their small area, distance to the nearest crossing, signals, bus stop, station, school, pub/bar and junction, and the type of road they happened on |
| `london_context.gpkg` → `osm_features`, `roads` | The OpenStreetMap features and roads used |
| `lsoa_variables.csv` | The small-area table without geometry, ready for modelling |

It ends with a borough ranking per resident and a first look at which small-area characteristics go together with serious pedestrian injuries.

### Step 3: hotspots and fair rates

```bash
python scripts/03_hotspots.py
```

Needs the street network from a full step 2 run. Three complementary views:

- **Street hotspots (network kernel density).** The drivable network is cut into 50 m pieces; each casualty is snapped to its street and its influence spreads along the network (quartic kernel, 250 m), so a cluster on one road does not leak onto a parallel street across a block.
- **Area hotspots (Getis-Ord Gi\*).** Small areas with significantly high or low casualty density compared with their neighbours, using 999 permutations and a false discovery rate check for multiple testing.
- **Fair rates.** Boroughs ranked per resident, per km² and per km of road, with Empirical Bayes smoothing of small-area rates so areas with few residents don't produce extreme values.

Outputs `data/processed/london_hotspots.gpkg` and the figures below, plus an interactive map in `figures/hotspots_map.html`.

![Street hotspots](figures/street_hotspots.png)

![Area hot and cold spots](figures/area_hotspots.png)

![Borough rankings change with the measure](figures/borough_rank_change.png)

### Step 4: the built environment model

```bash
python scripts/04_model.py          # main models, a few seconds
python scripts/04_model.py --gwr    # adds geographically weighted regression (20-60 minutes)
```

Negative binomial regression of pedestrian casualties per small area, with kilometres of road as the exposure, so effects are per km of road. Collision counts are over-dispersed (a few areas have very many), which is why negative binomial rather than Poisson is used; the script reports both for comparison.

- **Model A** uses activity only: population density, stations, bus stops, pubs and bars.
- **Model B** adds street design (major road share, junctions, crossings, signals, schools) and income deprivation.

Comparing them shows which street features still matter once an area's busyness is accounted for. Both are fitted for all pedestrian casualties and for fatal + serious ones, plus separately for inner and outer London.

Checks built in:

- **Collinearity:** variables with a variance inflation factor above 10 (mostly duplicating others) are dropped and reported.
- **Residual spatial pattern:** Moran's I on the model residuals.
- **Deprivation:** the IoD income score is used rather than the full IMD, which includes a road injury indicator.
- **Areas with more casualties than expected:** each small area is compared with what the model predicts for a place like it, with a Benjamini-Hochberg correction so that areas flagged are not just the extremes expected by chance among thousands.
- **Tested on synthetic data with known effects:** the model recovered every planted effect within its confidence interval, dropped a deliberately duplicated variable, flagged no areas when none were planted and found planted high-risk areas.

The optional geographically weighted Poisson regression maps where effects are stronger or weaker across London. It does not allow for over-dispersion, so it overstates significance; its maps are exploratory.

Results are written to `results/` (coefficients, full statistical output, flagged areas) and the figures below.

![Model effects](figures/model_effects.png)

![Inner vs outer London](figures/inner_outer_effects.png)

![Observed vs expected casualties](figures/excess_casualties.png)

**Interpreting the results.** These are associations across areas, not causes. Some street features are placed *because* of collisions (crossings and signals are often added after injuries), and area-level relationships do not necessarily hold for individual streets or people (the ecological fallacy). The value of the model is in showing which features go with more injuries than an area's busyness alone would predict, and where to look more closely.

### Step 5: access to emergency care

```bash
python scripts/05_access_to_care.py
```

Drive time along the street network from every pedestrian casualty to the nearest A&E department and to the nearest of London's four major trauma centres (The Royal London, St Mary's, King's College and St George's), where the most seriously injured are taken.

- **Trauma centres** are located from their postcodes (postcodes.io) and stored in `config/major_trauma_centres.csv`.
- **A&E departments** are hospitals tagged with an emergency department in OpenStreetMap, plus the trauma centres. The list used is saved to `results/ae_sites_from_osm.csv` for review; to correct it, copy it to `config/ae_sites.csv`, edit and re-run.
- **Travel times** use OpenStreetMap speed limits (typical speeds by road type where missing) on the strongly connected drivable network, with a multi-source shortest-path search from the hospitals over the reversed network.

Results: `results/access_by_borough.csv` and the figures below.

![Access to trauma care](figures/access_to_trauma_care.png)

![Share of serious casualties within X minutes](figures/access_curve.png)

**Limitations.** Times assume free-flowing traffic, so real journeys at busy times take longer. They cover the journey from the scene to hospital only, not the ambulance's journey to the scene, and ignore London's Air Ambulance and local trauma units. OpenStreetMap's emergency tags may be incomplete, which is why the A&E list can be reviewed and overridden.

### Step 6: interactive dashboard

```bash
python scripts/06_build_dashboard.py
```

Collects the results of steps 1–5, slims them for the web (shared borders simplified together so neighbouring areas still touch, coordinates rounded to about 1 m) and writes a self-contained site into `docs/`. Open `docs/index.html` to check it locally. Steps 4 and 5 are optional: their sections are left out if they haven't been run.

The dashboard has:

- a headline sentence that updates with the filters (severity, year, time of day, age, borough)
- casualties by hour of the day
- a **spotlight**: a circle (250 m, 500 m or 1 km) you can drag anywhere, which dims the rest of the map and summarises the casualties inside it (killed or seriously injured, children, by year, by hour, typical drive to trauma care, and the street stretches where serious injuries concentrate)
- **Play the years**: steps through 2021–2025, one year at a time, with the casualties of each year on the map
- four map views of London's small areas: hot and cold spots, more casualties than expected, casualties per km², and drive time to trauma care
- the most dangerous street stretches, individual casualties when zoomed in, and hospitals with A&E
- a borough panel with rankings three ways, typical drive time to trauma care and the streets where serious injuries concentrate
- the model's findings in plain language

**Publishing:** push to GitHub, then in the repository go to *Settings → Pages*, choose *Deploy from a branch*, branch `main`, folder `/docs`. The dashboard appears at `https://<your-username>.github.io/london-pedestrian-safety/` within a few minutes. Re-run the script and push again to update it.

### Step 7: Ask the data (AI assistant)

The dashboard's **Ask the data** box lets visitors ask questions in plain English, such as *"When are older pedestrians most at risk?"* or *"Serious injuries near Oxford Circus"*.

- **The AI translates; the data calculates.** A small relay (`ai/worker.js`, a free Cloudflare Worker) asks Claude Haiku 4.5 to turn the question into a structured query using tool calling. The dashboard validates every field against a fixed list, computes the numbers from the data in the browser, and moves the map, filters and spotlight to match.
- **Transparent:** every answer starts with "I read this as…" and restates exactly what was counted, with caveats where they matter (rates per resident, the 2021 lockdown, associations not causes).
- **Safe and cheap:** the API key stays in Cloudflare; the relay only serves this dashboard and limits use per visitor and per day; prepaid credit with auto-reload off means cost can't run away. Without it, visitors see a clear "currently unavailable" message and use the dashboard as normal.
- **Tested** with simulated Claude replies, including invalid and malicious ones, and every failure mode (no credit, busy, limits, network down).

Setup takes about 20 minutes in a browser: see [`ai/SETUP.md`](ai/SETUP.md).

## Data

| Dataset | Publisher | Licence |
|---|---|---|
| Road Safety Data (STATS19), last 5 published years | Department for Transport | Open Government Licence v3.0 |
| English Indices of Deprivation 2025 (File 7: scores, deciles, mid-2022 population) | Ministry of Housing, Communities and Local Government | Open Government Licence v3.0 |
| Lower layer Super Output Areas (December 2021) boundaries, generalised | Office for National Statistics | Open Government Licence v3.0 (contains OS data) |
| OpenStreetMap, Greater London extract and street network | OpenStreetMap contributors, via Geofabrik and OSMnx | Open Database Licence (ODbL) |
| Hospital locations (major trauma centres by postcode) | postcodes.io (ONS Postcode Directory) | Open Government Licence v3.0 |

### Known limitations

- STATS19 records injury collisions reported to the police. Minor pedestrian injuries are known to be under-reported.
- Some police forces moved to injury-based severity reporting, which shifts the balance between serious and slight injuries over time. DfT provides severity-adjusted figures, kept in the data, for comparisons across years.
- A small share of records have no usable location and are excluded from mapping (the script reports how many).
- Rates per resident overstate central London, where many more people walk than live. Later steps add measures that account for footfall.
- The Index of Multiple Deprivation has included a road traffic injury indicator (in its Living Environment domain). Using it to explain pedestrian injuries is therefore slightly circular; the effect is small, but step 4 also tests the income and employment scores on their own.
- OpenStreetMap is volunteer-mapped. London's coverage is very good, but crossings and signals may be less complete in some areas.

## Tools

Python, pandas, GeoPandas, OSMnx, PySAL (libpysal, esda, mgwr), SciPy and statsmodels for the analysis; Leaflet for the dashboard; Claude (Anthropic API) via a Cloudflare Worker for the AI assistant. The analysis uses only open data and open-source tools.
