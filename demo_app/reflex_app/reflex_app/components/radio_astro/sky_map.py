"""Sky map polar image component for the Radio Astronomy tab."""
import reflex as rx

from reflex_app.state.radio_astro import RadioAstroState


def radio_sky_map() -> rx.Component:
    return rx.vstack(
        rx.heading("Sky map at selected time", size="3"),
        rx.cond(
            RadioAstroState.sky_map_loading | (RadioAstroState.sky_map_base64 == ""),
            rx.vstack(
                rx.spinner(size="3"),
                rx.text("Computing sky map...", color_scheme="gray", size="2"),
                spacing="2",
                align_items="center",
                padding_y="40px",
            ),
            rx.image(
                src=RadioAstroState.sky_map_base64,
                width="100%",
                max_width="500px",
            ),
        ),
        spacing="2",
        width="100%",
        align_items="center",
    )
