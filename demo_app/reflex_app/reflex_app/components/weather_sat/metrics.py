"""Metric cards and peak RFI table for the Weather Satellite tab."""
import reflex as rx

from reflex_app.state.weather_sat import WeatherSatState
from reflex_app.utils.weather_loaders_reflex import NEGLIGIBLE_RFI_DBW


def weather_metrics() -> rx.Component:
    m = WeatherSatState.metrics

    return rx.vstack(
        # Starlink metrics row (4 cards)
        rx.text("Starlink", weight="bold", size="3"),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.text("Visible Starlinks", size="1", color_scheme="gray"),
                    rx.heading(m.get("n_starlinks", 0).to_string(), size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Fundamental", size="1", color_scheme="gray"),
                    rx.heading(m.get("starlink_freq_ghz", 0.0).to_string() + " GHz", size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("RFI K-Band [dBW]", size="1", color_scheme="gray"),
                    rx.heading(m.get("k_starlink_dbw", 0.0).to_string(), size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("RFI V-Band [dBW]", size="1", color_scheme="gray"),
                    rx.heading(m.get("v_starlink_dbw", 0.0).to_string(), size="5"),
                ),
            ),
            spacing="3",
            wrap="wrap",
            width="100%",
        ),

        # 5G metrics row (5 cards)
        rx.text("5G cellular network (mmWave)", weight="bold", size="3"),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.text("Emitters in FOV", size="1", color_scheme="gray"),
                    rx.heading(m.get("n_emitters", 0).to_string(), size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("Fundamental", size="1", color_scheme="gray"),
                    rx.heading(m.get("five_g_freq_ghz", 0.0).to_string() + " GHz", size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("EIRP", size="1", color_scheme="gray"),
                    rx.heading(m.get("five_g_eirp_dbw", 0.0).to_string() + " dBW", size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("RFI K-Band [dBW]", size="1", color_scheme="gray"),
                    rx.heading(m.get("k_5g_dbw", 0.0).to_string(), size="5"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.text("RFI V-Band [dBW]", size="1", color_scheme="gray"),
                    rx.heading(m.get("v_5g_dbw", 0.0).to_string(), size="5"),
                ),
            ),
            spacing="3",
            wrap="wrap",
            width="100%",
        ),

        rx.text(
            "ATMS: K 23.8 GHz / 270 MHz, V 50.3 GHz / 180 MHz. "
            "5G uses equivalent emitter at FOV center × emitter count.",
            size="1", color_scheme="gray",
        ),

        spacing="3",
        width="100%",
    )


def weather_peak_table() -> rx.Component:
    p = WeatherSatState.peak_rfi

    return rx.vstack(
        rx.text("Peak RFI over overpass", weight="bold", size="3"),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Source"),
                    rx.table.column_header_cell("Max K-Band (23.8 GHz) [dBW]"),
                    rx.table.column_header_cell("Max V-Band (50.3 GHz) [dBW]"),
                )
            ),
            rx.table.body(
                rx.table.row(
                    rx.table.cell("Starlink"),
                    rx.table.cell(p.get("k_starlink", 0.0).to_string()),
                    rx.table.cell(p.get("v_starlink", 0.0).to_string()),
                ),
                rx.table.row(
                    rx.table.cell("5G"),
                    rx.table.cell(p.get("k_5g", 0.0).to_string()),
                    rx.table.cell(p.get("v_5g", 0.0).to_string()),
                ),
            ),
            width="100%",
        ),
        rx.text(
            f"Peak = max over overpass (10 s grid). Negligible RFI shown as "
            f"{NEGLIGIBLE_RFI_DBW:.0f} dBW (not −1000 dBW).",
            size="1", color_scheme="gray",
        ),
        spacing="2",
        width="100%",
    )
