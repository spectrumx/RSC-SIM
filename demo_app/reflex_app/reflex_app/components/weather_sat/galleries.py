"""Antenna pattern gallery for the Weather Satellite tab."""
import reflex as rx


def weather_galleries() -> rx.Component:
    return rx.accordion.root(
        rx.accordion.item(
            header="Antenna patterns",
            content=rx.vstack(
                rx.hstack(
                    rx.image(
                        src="/gallery/starlink_antenna_pattern.png",
                        width="50%",
                    ),
                    rx.image(
                        src="/gallery/ground_emitter_5g_antenna_pattern.png",
                        width="50%",
                    ),
                    width="100%",
                ),
                rx.image(
                    src="/gallery/weather_sat_antenna_patterns.png",
                    width="100%",
                ),
                spacing="2",
            ),
            value="antenna",
        ),
        rx.accordion.item(
            header="Satellite positions",
            content=rx.image(
                src="/gallery/satellite_positions.png",
                width="100%",
            ),
            value="satpos",
        ),
        type="multiple",
        width="100%",
    )
