"""Time series Plotly chart for the Radio Astronomy tab."""
import plotly.graph_objects as go
import reflex as rx

from reflex_app.state.radio_astro import RadioAstroState


def radio_time_series() -> rx.Component:
    return rx.vstack(
        rx.heading("Received power vs time", size="3"),
        rx.cond(
            RadioAstroState.loading,
            rx.spinner(size="3"),
            rx.cond(
                RadioAstroState.time_series_fig != {},
                rx.plotly(
                    data=RadioAstroState.time_series_fig.to(go.Figure),
                    width="100%",
                ),
                rx.text("Run a simulation to see the time series.", color_scheme="gray"),
            ),
        ),
        rx.cond(
            RadioAstroState.constellation_enabled,
            rx.text(
                "Green = clean observation; blue = Starlink on (no avoidance); "
                "orange dotted = beam avoidance. "
                "Shaded bands: OFF-source (first 5 min) then ON-source (Cas A tracked).",
                size="1", color_scheme="gray",
            ),
            rx.text(
                "Green = clean observation (Starlink off). "
                "Shaded bands: OFF-source (first 5 min) then ON-source (Cas A tracked).",
                size="1", color_scheme="gray",
            ),
        ),
        spacing="2",
        width="100%",
    )
