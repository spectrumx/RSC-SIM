"""RSC-SIM Reflex Demo App — entry point."""

from rxconfig import config

import reflex as rx

from reflex_app.pages.index import index
from reflex_app.state.radio_astro import RadioAstroState
from reflex_app.state.weather_sat import WeatherSatState


app = rx.App(
    theme=rx.theme(appearance="dark", accent_color="blue"),
)

app.add_page(
    index,
    route="/",
    title="RSC-SIM Demo",
    on_load=[
        RadioAstroState.load_resources,
        WeatherSatState.load_resources,
    ],
)
