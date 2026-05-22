"""
Tab 3: Pre-baked scenario gallery.

Displays images / data rendered ahead of time by `demo_app/precompute.py`.
Useful for two reasons:

1. Heavy NWP RFI runs (`ATMS_RFI_modeling.py` etc.) take minutes to hours --
   not live-demoable. Their outputs are baked into the gallery.
2. If the live tabs hiccup at the booth, the gallery still works because all
   it does is read PNGs from disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

import sim_cache  # noqa: F401  (registers `src/` on sys.path)


_GALLERY_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assets" / "gallery"


_ITEM_TITLES = {
    "doppler": "Doppler waterfall",
    "weather_summary": "Weather sat antennas (K-Band vs V-Band)",
    "nwp": "NWP TMBR vs TMBR_RFI",
}


_ITEM_CAPTIONS = {
    "doppler": (
        "A Starlink carrier sweeping across the radio receiver band as the "
        "satellite passes overhead. The streak is the Doppler-shifted carrier; "
        "exactly the kind of feature that confuses automatic RFI flaggers. "
        "Built from the bundled 2025-02-18 Starlink trajectory."
    ),
    "weather_summary": (
        "Suomi-NPP K-Band (23.8 GHz) and V-Band (50.3 GHz) antenna patterns. "
        "Starlink fundamental at ~11.9 GHz puts its 2nd harmonic into K-Band "
        "and its 4th into V-Band, both arriving through these sidelobes."
    ),
    "nwp": (
        "Native TMBR vs TMBR + RFI from the NWP pipeline output. The right-hand "
        "panel is what an operational data assimilation system would see if "
        "Starlink + 5G mitigation isn't applied."
    ),
}


def _read_manifest() -> dict:
    manifest_path = _GALLERY_DIR / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def render() -> None:
    st.markdown(
        """
        ### Pre-baked scenario gallery
        Heavy or pre-rendered analyses live here: NWP brightness-temperature
        maps, Doppler waterfalls, and other artifacts that aren't suitable
        for live recompute. Run `python demo_app/precompute.py` from the repo
        root to (re)generate these images.
        """
    )

    manifest = _read_manifest()
    if not manifest:
        st.warning(
            "No gallery manifest found. From the repo root run "
            "`python demo_app/precompute.py` to render the assets."
        )
        return

    items = manifest.get("items", {})
    available = [(k, v) for k, v in items.items() if v]
    if not available:
        st.info("Gallery manifest is empty. Re-run `python demo_app/precompute.py`.")
        return

    st.caption(f"Last rendered: {manifest.get('generated_at', 'unknown')}")

    for key, filename in available:
        title = _ITEM_TITLES.get(key, key)
        caption = _ITEM_CAPTIONS.get(key, "")
        st.markdown(f"#### {title}")
        if caption:
            st.markdown(caption)
        path = _GALLERY_DIR / filename
        if path.is_file():
            st.image(str(path), width="stretch")
        else:
            st.warning(f"Asset `{filename}` was registered but is missing on disk.")
        st.markdown("---")
