"""Ground tracks Plotly Scattergeo for the Radio Astronomy tab.

The ground track figure dict is built in the state's recompute_sky_map()
background task and stored in ground_track_fig (dict field). The component
just renders it.
"""
import plotly.graph_objects as go
import reflex as rx

from reflex_app.state.radio_astro import RadioAstroState


def radio_ground_tracks() -> rx.Component:
    return rx.accordion.root(
        rx.accordion.item(
            header="Satellite ground tracks (Earth view)",
            content=rx.cond(
                RadioAstroState.constellation_enabled,
                rx.cond(
                    RadioAstroState.sky_map_loading,
                    rx.text("Computing ground tracks...", color_scheme="gray", size="2"),
                    rx.cond(
                        RadioAstroState.ground_track_fig != {},
                        rx.plotly(
                            data=RadioAstroState.ground_track_fig.to(go.Figure),
                            width="100%",
                        ),
                        rx.text(
                            "No satellite tracks for current selection. "
                            "Click 'Generate Sky Map' to compute.",
                            color_scheme="gray",
                        ),
                    ),
                ),
                rx.text(
                    "Enable 'Include Starlink constellation' to view ground tracks.",
                    color_scheme="gray",
                ),
            ),
            value="tracks",
        ),
        default_value=["tracks"],
        type="multiple",
        width="100%",
    )
