"""Main page: tabbed layout for both demo scenarios."""
import reflex as rx

from reflex_app.components.common.header import app_header
from reflex_app.components.radio_astro.controls import radio_controls
from reflex_app.components.radio_astro.metrics import radio_metrics
from reflex_app.components.radio_astro.time_series import radio_time_series
from reflex_app.components.radio_astro.sky_map import radio_sky_map
from reflex_app.components.radio_astro.ground_tracks import radio_ground_tracks
from reflex_app.components.weather_sat.controls import weather_controls
from reflex_app.components.weather_sat.metrics import weather_metrics, weather_peak_table
from reflex_app.components.weather_sat.time_series import weather_time_series
from reflex_app.components.weather_sat.fov_map import weather_fov_map
from reflex_app.components.weather_sat.galleries import weather_galleries
from reflex_app.state.radio_astro import RadioAstroState
from reflex_app.state.weather_sat import WeatherSatState

# Explicit pixel margin applied consistently to all page sections
_PX = "56px"


def _loading_overlay(message: str = "Loading resources...") -> rx.Component:
    return rx.vstack(
        rx.spinner(size="3"),
        rx.text(message, color_scheme="gray", size="2"),
        spacing="3",
        align_items="center",
        justify_content="center",
        padding_y="80px",
        width="100%",
    )


def _radio_tab_content() -> rx.Component:
    """Radio astronomy tab: narrow controls column + wide plots area."""
    return rx.cond(
        RadioAstroState.resources_ready,
        rx.flex(
            # Left: controls column — full-width on mobile, fixed 280px on desktop
            rx.box(
                radio_controls(),
                width={"base": "100%", "md": "280px"},
                min_width={"base": "0", "md": "280px"},
                flex_shrink="0",
                overflow_y="auto",
            ),
            # Right: main panel fills remaining space
            rx.box(
                rx.vstack(
                    rx.cond(
                        RadioAstroState.load_error != "",
                        rx.callout(
                            RadioAstroState.load_error,
                            color_scheme="red",
                            width="100%",
                        ),
                        rx.fragment(),
                    ),
                    radio_metrics(),
                    # Sky map + time series: side by side on desktop, stacked on mobile
                    rx.flex(
                        rx.box(radio_sky_map(), flex="1", min_width="0"),
                        rx.box(radio_time_series(), flex="1", min_width="0"),
                        flex_wrap="wrap",
                        gap="4",
                        width="100%",
                        align_items="flex_start",
                    ),
                    radio_ground_tracks(),
                    spacing="4",
                    width="100%",
                    align_items="stretch",
                ),
                flex="1",
                min_width="0",
                padding="4",
                overflow_x="hidden",
            ),
            flex_wrap={"base": "wrap", "md": "nowrap"},
            padding_x=_PX,
            width="100%",
            align_items="flex_start",
            gap="24px",
        ),
        rx.box(
            _loading_overlay("Loading radio astronomy resources…"),
            padding_x=_PX,
            width="100%",
        ),
    )


def _weather_tab_content() -> rx.Component:
    """Weather satellite tab: narrow controls column + wide plots area."""
    return rx.cond(
        WeatherSatState.resources_ready,
        rx.flex(
            # Left: controls column — full-width on mobile, fixed 280px on desktop
            rx.box(
                weather_controls(),
                width={"base": "100%", "md": "280px"},
                min_width={"base": "0", "md": "280px"},
                flex_shrink="0",
                overflow_y="auto",
            ),
            # Right: main panel fills remaining space
            rx.box(
                rx.vstack(
                    rx.cond(
                        WeatherSatState.load_error != "",
                        rx.callout(
                            WeatherSatState.load_error,
                            color_scheme="red",
                            width="100%",
                        ),
                        rx.fragment(),
                    ),
                    weather_metrics(),
                    weather_peak_table(),
                    weather_time_series(),
                    weather_fov_map(),
                    weather_galleries(),
                    spacing="4",
                    width="100%",
                    align_items="stretch",
                ),
                flex="1",
                min_width="0",
                padding="4",
                overflow_x="hidden",
            ),
            flex_wrap={"base": "wrap", "md": "nowrap"},
            padding_x=_PX,
            width="100%",
            align_items="flex_start",
            gap="24px",
        ),
        rx.box(
            _loading_overlay("Loading weather satellite resources…"),
            padding_x=_PX,
            width="100%",
        ),
    )


def index() -> rx.Component:
    """Root page component."""
    return rx.box(
        rx.box(
            app_header(),
            padding_x=_PX,
            padding_top="40px",
            padding_bottom="2",
        ),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Starlink vs. radio telescope", value="radio"),
                rx.tabs.trigger("Weather satellite FOV", value="weather"),
                padding_x=_PX,
            ),
            rx.tabs.content(
                rx.vstack(
                    rx.markdown(
                        "### Starlink vs. radio telescope (Looking-Up)\n"
                        "Live RFI sandbox using the bundled **Westford / Cas A / Starlink** data "
                        "(2025-02-18). Toggle the constellation, dial in beam avoidance, or pick "
                        "a single satellite for forensic mode.",
                        padding_x=_PX,
                        padding_y="2",
                    ),
                    _radio_tab_content(),
                    spacing="0",
                    width="100%",
                ),
                value="radio",
                width="100%",
            ),
            rx.tabs.content(
                rx.vstack(
                    rx.markdown(
                        "### Weather satellite single FOV (Looking-Down)\n"
                        "Suomi-NPP / JPSS ATMS **K-Band (23.8 GHz)** and **V-Band (50.3 GHz)** RFI "
                        "from Starlink back/side lobes and **5G mmWave** ground emitters over a "
                        "single FOV circle of 32 km diameter at Westford.",
                        padding_x=_PX,
                        padding_y="2",
                    ),
                    _weather_tab_content(),
                    spacing="0",
                    width="100%",
                ),
                value="weather",
                width="100%",
            ),
            default_value="radio",
            width="100%",
        ),
        width="100%",
        min_height="100vh",
    )
