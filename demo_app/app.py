"""
RSC-SIM Conference Demo App -- Streamlit entry point.

Run with:
    streamlit run demo_app/app.py

The app exposes three tabs:

1. Starlink vs. radio telescope (live, headline)
2. Weather satellite single FOV (live, Phase 1)
3. Pre-baked scenario gallery (fallback, requires `precompute.py`)

This entry-point is intentionally thin: each tab lives in its own module under
`demo_app/panels/`, and `sim_cache` owns all heavy resource loading.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Make `demo_app/` importable so the panel modules can `import sim_cache`
# without relying on package semantics. Streamlit runs this script as
# `__main__`, so we must do the path setup ourselves.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# `sim_cache` further appends `src/` and `educational_tutorials/` to `sys.path`.
import sim_cache  # noqa: E402
from panels import gallery, radio_astro, weather_fov  # noqa: E402


def _inject_layout_css() -> None:
    """Tighten Streamlit's default top padding above the page title."""
    st.markdown(
        """
        <style>
        div.block-container {
            padding-top: 1rem;
        }
        section.main > div.block-container {
            padding-top: 1rem;
        }
        .stMainBlockContainer {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _configure_page() -> None:
    st.set_page_config(
        page_title="RSC-SIM Demo",
        page_icon=None,
        layout="wide",
        # Each tab carries its own controls column, so the global sidebar
        # starts collapsed (and is empty by default).
        initial_sidebar_state="collapsed",
    )
    _inject_layout_css()


def _global_header() -> None:
    st.title("RSC-SIM — Radio Science Coexistence Simulator")
    st.markdown(
        """
        Live demonstrations of how satellite mega-constellations and terrestrial
        emitters affect scientific observations, built on the
        [RSC-SIM](https://github.com/spectrumx/RSC-SIM) framework.

        Each tab has its own **Controls** column on the left for tweaking
        parameters; plots update in real time. Each tab also includes a
        *Reset* button for booth visitors.
        """
    )


def main() -> None:
    _configure_page()
    _global_header()

    # Pre-warm caches so the first slider drag is fast. This call is wrapped
    # in a Streamlit spinner inside the cache loaders themselves, so the user
    # sees progress on cold starts.
    with st.status("Pre-warming caches...", expanded=False) as status:
        try:
            sim_cache.prewarm_radio_astro()
            status.update(label="Caches ready.", state="complete")
        except Exception as exc:  # pragma: no cover - booth safety net
            status.update(
                label=f"Cache pre-warm failed: {exc}", state="error", expanded=True
            )

    tabs = st.tabs(
        [
            "Starlink vs. radio telescope",
            "Weather satellite FOV",
            "Pre-baked gallery",
        ]
    )

    with tabs[0]:
        radio_astro.render()
    with tabs[1]:
        weather_fov.render()
    with tabs[2]:
        gallery.render()


if __name__ == "__main__":
    main()
