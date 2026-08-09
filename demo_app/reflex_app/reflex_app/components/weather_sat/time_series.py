"""Dual-subplot K/V band time series for the Weather Satellite tab."""
import plotly.graph_objects as go
import reflex as rx

from reflex_app.state.weather_sat import WeatherSatState


def weather_time_series() -> rx.Component:
    return rx.vstack(
        rx.heading("RFI power vs time", size="3"),
        rx.cond(
            WeatherSatState.loading,
            rx.spinner(size="3"),
            rx.cond(
                WeatherSatState.time_series_fig != {},
                rx.plotly(
                    data=WeatherSatState.time_series_fig.to(go.Figure),
                    width="100%",
                ),
                rx.text("Loading...", color_scheme="gray"),
            ),
        ),
        rx.text(
            "Demo simplifications: 10 s time grid, no DEM terrain masking, "
            "Starlink elevation > 0° (no DTC), atmospheric loss disabled, "
            "5G = one link budget at FOV center × n_emitters. OOBE not modeled.",
            size="1", color_scheme="gray",
        ),
        spacing="2",
        width="100%",
    )
