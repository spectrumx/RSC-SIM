"""FOV ground footprint Scattergeo for the Weather Satellite tab."""
import plotly.graph_objects as go
import reflex as rx

from reflex_app.utils.weather_loaders_reflex import TARGET_LAT, TARGET_LON


def _fov_fig_dict() -> dict:
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=[TARGET_LAT],
        lon=[TARGET_LON],
        text=["Westford FOV center (32 km)"],
        mode="markers+text",
        marker=dict(size=18, color="#1f77b4", symbol="circle"),
        name="FOV center",
    ))
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(50, 50, 50)",
            showocean=True,
            oceancolor="rgb(20, 20, 30)",
            showcountries=True,
            countrycolor="rgb(80, 80, 80)",
            projection_type="natural earth",
            center=dict(lat=TARGET_LAT, lon=TARGET_LON),
            projection_scale=8,
        ),
        title="FOV ground footprint",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig.to_dict()


_FOV_FIG = go.Figure(_fov_fig_dict())


def weather_fov_map() -> rx.Component:
    return rx.vstack(
        rx.heading("FOV ground footprint (Earth view)", size="3"),
        rx.plotly(data=_FOV_FIG, width="100%"),
        spacing="2",
        width="100%",
    )
