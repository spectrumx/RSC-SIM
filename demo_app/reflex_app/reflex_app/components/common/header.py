"""App-level header component."""
import reflex as rx


def app_header() -> rx.Component:
    return rx.vstack(
        rx.heading("RSC-SIM — Radio Science Coexistence Simulator", size="7"),
        rx.text(
            "Live demonstrations of how satellite mega-constellations and terrestrial "
            "emitters affect scientific observations, built on the RSC-SIM framework.",
            color_scheme="gray",
            size="2",
        ),
        spacing="1",
        width="100%",
        padding_bottom="4",
    )
