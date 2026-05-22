# RSC-SIM Conference Demo App

A live, interactive Streamlit dashboard for showcasing **RSC-SIM** at conferences and demonstrations. It wraps the existing simulation modules in a single laptop-friendly UI so booth visitors can tweak parameters (time, frequency, beam avoidance, satellite count, gateway location, etc.) and see updated sky maps, time series, and Earth maps in seconds.

## What's in here

```
demo_app/
  app.py                 # Streamlit entry point with three tabs
  sim_cache.py           # Cached loaders (Antenna, Trajectory, Constellation, sky models)
  precompute.py          # Optional: pre-render scenario PNGs / NPZ files for the gallery
  panels/
    radio_astro.py       # Tab 1: Starlink vs. radio telescope (live)
    weather_fov.py       # Tab 2: Weather satellite single FOV (live, Phase 1)
    gallery.py           # Tab 3: Pre-baked NWP / Doppler scenarios (fallback)
  assets/
    narrative.md         # Booth-friendly scripts and captions
  README.md              # This file
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
- Beam-avoidance angle (0 deg = off)
- Number of satellites included (subset of the Starlink trajectory)
- Direct-mode dropdown to mirror `tuto_radiomdl_direct.py`

Live outputs (Looking-Up case — power in **dBW**, matching RSC-SIM tutorials):

- Polar sky map with telescope gain in dBW, satellite markers, and Cas A
- Plotly received-power time-series (dBW) with shaded interference events
- Earth map of satellite ground tracks (pydeck)
- Headline tile: peak power [dBW], peak excess [dB], fraction of time over a threshold

### 2. Weather satellite single FOV

Live (Phase 1) version of `research_tutorials/tuto_radiomdl_weather_phase1.py`:

- Sensor band: K-Band 23.8 GHz vs. V-Band 50.3 GHz
- 5G emitter density slider (Phase 2 hooks - shown only when phase data is staged)
- Starlink gateway location pickable on a Folium map (Phase 2)
- Time scrubber within a JPSS overpass

Outputs include the FOV ground footprint, gateway location, and stacked Tb contributions.

### 3. Pre-baked gallery

Fallback for heavy NWP runs and the Doppler waterfall. Reads pre-rendered PNGs / numpy archives under `demo_app/assets/`. Generate them once with:

```bash
python demo_app/precompute.py
```

## Booth checklist

Before opening the booth:

1. Pre-warm the cache: launch the app, click through each tab once, drag a slider so the simulation runs.
2. Verify offline: disable Wi-Fi and re-load the page. Should still work; only Streamlit's "version available" toast may complain.
3. Confirm bundled data files exist in `research_tutorials/data/`:
   - `single_cut_res.cut`
   - `casA_trajectory_Westford_*.arrow`
   - `Starlink_trajectory_Westford_*.arrow`
   - `jpss_trajectory_Westford_*.arrow` (for the weather tab)
   - `K-Band 23.8 GHz absolute antenna pattern.csv`, `V-Band 50.3 GHz absolute antenna pattern.csv`
4. Stretch / heavy data (only if running gallery from scratch):
   - `GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif`
   - `itu_iclw_rain_info_MM.nc`
5. Pre-render gallery scenarios with `python demo_app/precompute.py` (offline-safe).
6. Practice the 60-second narrative under `demo_app/assets/narrative.md`.

## Troubleshooting

- **Slow first render**: expected. The Antenna and Trajectory loaders deserialize Arrow / `.cut` files. They are cached after first call (`@st.cache_resource`).
- **`ModuleNotFoundError: streamlit`**: run `pip install -e ".[demo]"` from the repo root.
- **`FileNotFoundError` for Cas A / Starlink arrow**: the bundled `2025-02-18` files must be in `research_tutorials/data/`. Verify with `ls research_tutorials/data/`.
- **Plots look empty after dragging a slider**: most likely the elevation filter excluded all points. Bump up the time scrubber or lower the minimum-elevation threshold.

## License

MIT. See repo-level `LICENSE` (or `pyproject.toml`).
