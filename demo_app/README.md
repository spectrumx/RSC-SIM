# RSC-SIM Conference Demo App

A live, interactive Streamlit dashboard for showcasing **RSC-SIM** at conferences and demonstrations. It wraps the existing simulation modules in a single laptop-friendly UI so booth visitors can tweak parameters (time, frequency, beam avoidance, satellite count, gateway location, etc.) and see updated sky maps, time series, and Earth maps in seconds.

## What's in here

```
demo_app/
  app.py                       # Streamlit entry point with two tabs
  sim_cache.py                 # Cached loaders (Antenna, Trajectory, Constellation, sky models)
  panels/
    radio_astro.py             # Tab 1: Starlink vs. radio telescope (live)
    weather_fov.py             # Tab 2: Weather satellite single FOV (live, Phase 2 demo)
    weather_phase2_loaders.py  # Cached Phase-2 RFI loaders for Tab 2
  assets/
    narrative.md               # Booth-friendly scripts and captions
  README.md                    # This file
```

## Quick start

From the **repo root**:

```bash
# Create a virtual environment (recommended)
python -m venv rsc-sim-env
source rsc-sim-env/bin/activate  # Windows: rsc-sim-env\Scripts\activate

# Install RSC-SIM plus the demo dependencies (Streamlit, Plotly, pydeck, Folium)
pip install -e ".[demo]"

# Run the app
streamlit run demo_app/app.py
```

Streamlit will print a `localhost` URL (default `http://localhost:8501`). Open it in any browser. The first launch loads the trajectory and antenna files into a process-wide cache, so the first slider drag may take a few seconds; subsequent updates are sub-second.

### Run offline at the booth

`demo_app/.streamlit/config.toml` already sets `headless = true`,
`gatherUsageStats = false`, and a dark theme. Streamlit auto-loads it when you
launch the app from inside `demo_app/`:

```bash
cd demo_app && streamlit run app.py
```

If you launch from the repo root instead, pass the same flags explicitly:

```bash
streamlit run demo_app/app.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8501
```

## Demo tabs at a glance

### 1. Starlink vs. radio telescope (headline)

Live sandbox built on `educational_tutorials/02_satellite_interference.py` and `educational_tutorials/03_sky_mapping.py`. All data ships with the repo (`research_tutorials/data/`).

Tweakable controls (left **Controls** column, in-tab):

- Time scrubber within the 45-minute Cas A pass
- Center frequency and bandwidth (clamped to the antenna's valid band)
- **Include Starlink constellation** toggle (off by default for a clean start)
- Beam-avoidance angle (0 deg = off; disabled until constellation is on)
- Number of satellites included (0 = all; disabled until constellation is on)
- Direct-mode dropdown to mirror `tuto_radiomdl_direct.py` (disabled until constellation is on)

Live outputs (Looking-Up case — power in **dBW**, matching RSC-SIM tutorials):

- Polar sky map with telescope gain in dBW, satellite markers, and Cas A
- Plotly received-power time-series (dBW) with shaded interference events
- Earth map of satellite ground tracks (pydeck)
- Headline tiles: visible satellite count and peak received power [dBW]

### 2. Weather satellite single FOV

Live Phase-2 demo based on `research_tutorials/tuto_radiomdl_weather_phase2.py` (Suomi-NPP / JPSS ATMS, Westford 32 km FOV, 2025-11-01 overpass):

- **Both** K-Band (23.8 GHz) and V-Band (50.3 GHz) RFI time series on one chart
- Starlink back/side-lobe RFI (full phase-2 ECEF model) + **5G mmWave** via equivalent emitter at FOV center
- Starlink controls: fundamental freq (10.7–12.7 GHz), EIRP
- 5G controls: mmWave fundamental freq (23.8–50.3 GHz), EIRP (−8.5 to 40 dBW), emitter density (1–50 / km²)
- Time scrubber adds a marker on the series (does not recompute on drag)

Live outputs (RFI power in **dBW**):

- Instantaneous metrics for Starlink and 5G (visible Starlinks, fundamentals, K/V RFI at selected time)
- Peak RFI table (Starlink / 5G × K / V max over the overpass)
- Plotly dual-subplot RFI time series with vertical time marker
- Collapsible **Antenna patterns** and **Satellite positions** PNGs under `demo_app/assets/gallery/` (optional)
- FOV ground footprint map (Westford center)

Demo simplifications: 10 s time grid, no DEM, Starlink elev filter at 0° (no DTC; bundled trajectory already uses a ~5° pass mask at creation), atmospheric loss disabled, 2nd harmonic factor 0.01 (−20 dBc), 3 dB polarization loss for Starlink, negligible RFI shown as −500 dBW.

Optional gallery PNGs: `starlink_antenna_pattern.png`, `ground_emitter_5g_antenna_pattern.png`, `weather_sat_antenna_patterns.png`, `satellite_positions.png`.

## Booth checklist

Before opening the booth:

1. Pre-warm the cache: launch the app, click through each tab once, drag a slider so the simulation runs.
2. Verify offline: disable Wi-Fi and re-load the page. Should still work; only Streamlit's "version available" toast may complain.
3. Confirm bundled data files exist in `research_tutorials/data/`:
   - **Tab 1:** `single_cut_res.cut`, `casA_trajectory_Westford_2025-02-18*.arrow`, `Starlink_trajectory_Westford_2025-02-18*.arrow`
   - **Tab 2:** `jpss_trajectory_Westford_2025-11-01*.arrow`, `Starlink_trajectory_Westford_2025-11-01*.arrow`, `K-Band 23.8 GHz absolute antenna pattern.csv`, `V-Band 50.3 GHz absolute antenna pattern.csv`
4. Practice the 60-second narrative under `demo_app/assets/narrative.md`.

## Troubleshooting

- **Slow first render**: expected. The Antenna and Trajectory loaders deserialize Arrow / `.cut` files. They are cached after first call (`@st.cache_resource`).
- **`ModuleNotFoundError: streamlit`**: run `pip install -e ".[demo]"` from the repo root.
- **`FileNotFoundError` for Cas A / Starlink arrow**: the bundled `2025-02-18` files must be in `research_tutorials/data/`. Verify with `ls research_tutorials/data/`.
- **Plots look empty after dragging a slider**: most likely the elevation filter excluded all points. Bump up the time scrubber or lower the minimum-elevation threshold.

## License

MIT. See repo-level `LICENSE` (or `pyproject.toml`).
