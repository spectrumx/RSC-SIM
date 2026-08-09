"""Metric cards for the Radio Astronomy tab."""
import reflex as rx

from reflex_app.state.radio_astro import RadioAstroState


def radio_metrics() -> rx.Component:
    return rx.hstack(
        rx.card(
            rx.vstack(
                rx.text("Visible satellites", size="1", color_scheme="gray"),
                rx.heading(
                    RadioAstroState.metrics.get("visible_sats", 0).to_string(),
                    size="6",
                ),
            ),
        ),
        rx.card(
            rx.vstack(
                rx.text("Peak power [dBW]", size="1", color_scheme="gray"),
                rx.heading(
                    RadioAstroState.metrics.get("peak_power_dbw", 0.0).to_string(),
                    size="6",
                ),
            ),
        ),
        spacing="4",
        width="100%",
    )
