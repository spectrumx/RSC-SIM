"""Controls panel for the Weather Satellite tab."""
import reflex as rx

from reflex_app.state.weather_sat import WeatherSatState


def _slider_row(
    label: str,
    min_val,
    max_val,
    step,
    default_value,
    on_change,
    on_value_commit,
    display_value,
    slider_key=None,
    disabled=None,
) -> rx.Component:
    """Slider with label, live value readout, and min/max tick labels."""
    label_row = rx.hstack(
        rx.text(label, size="1", weight="bold"),
        rx.spacer(),
        rx.text(display_value, size="1", color_scheme="blue", weight="bold"),
        width="100%",
    )
    slider_kwargs = dict(
        min=min_val,
        max=max_val,
        step=step,
        default_value=default_value,
        on_change=on_change,
        on_value_commit=on_value_commit,
        width="100%",
    )
    if disabled is not None:
        slider_kwargs["disabled"] = disabled
    if slider_key is not None:
        slider_kwargs["key"] = slider_key

    minmax_row = rx.hstack(
        rx.text(min_val, size="1", color_scheme="gray"),
        rx.spacer(),
        rx.text(max_val, size="1", color_scheme="gray"),
        width="100%",
    )
    return rx.vstack(
        label_row,
        rx.slider(**slider_kwargs),
        minmax_row,
        spacing="1",
        width="100%",
    )


def weather_controls() -> rx.Component:
    rk = WeatherSatState.reset_key.to_string()

    return rx.vstack(
        rx.heading("Controls", size="4"),

        rx.button(
            "Reset",
            on_click=WeatherSatState.reset_to_defaults,
            width="100%",
            variant="outline",
        ),

        # --- Time scrubber ---
        rx.accordion.root(
            rx.accordion.item(
                header="Time",
                content=rx.vstack(
                    rx.text(
                        "(moves marker, no recompute)",
                        size="1", color_scheme="gray",
                    ),
                    _slider_row(
                        label="Minutes from start of overpass",
                        min_val=0.0,
                        max_val=WeatherSatState.obs_total_minutes,
                        step=0.25,
                        default_value=[1.0],
                        on_change=WeatherSatState.preview_time_offset,
                        on_value_commit=WeatherSatState.set_time_offset,
                        display_value=WeatherSatState.display_time_offset_min.to_string() + " min",
                        slider_key="wx-time-" + rk,
                    ),
                    spacing="2",
                ),
                value="time",
            ),
            default_value=["time"],
            type="multiple",
            width="100%",
        ),

        # --- Starlink ---
        rx.accordion.root(
            rx.accordion.item(
                header="Starlink Interference",
                content=rx.vstack(
                    rx.text(
                        "11.9→2nd harm K-band; 12.575→4th harm V-band",
                        size="1", color_scheme="gray",
                    ),
                    _slider_row(
                        label="Fundamental freq [GHz]",
                        min_val=10.7,
                        max_val=12.7,
                        step=0.005,
                        default_value=[11.9],
                        on_change=WeatherSatState.preview_starlink_freq,
                        on_value_commit=WeatherSatState.set_starlink_freq,
                        display_value=WeatherSatState.display_starlink_freq_ghz.to_string() + " GHz",
                        slider_key="wx-slfreq-" + rk,
                    ),
                    _slider_row(
                        label="EIRP [dBW]",
                        min_val=-50.0,
                        max_val=20.0,
                        step=0.5,
                        default_value=[10.0],
                        on_change=WeatherSatState.preview_starlink_eirp,
                        on_value_commit=WeatherSatState.set_starlink_eirp,
                        display_value=WeatherSatState.display_starlink_eirp_dbw.to_string() + " dBW",
                        slider_key="wx-sleirp-" + rk,
                    ),
                    spacing="2",
                ),
                value="starlink",
            ),
            default_value=["starlink"],
            type="multiple",
            width="100%",
        ),

        # --- 5G ---
        rx.accordion.root(
            rx.accordion.item(
                header="5G Interference",
                content=rx.vstack(
                    _slider_row(
                        label="Fundamental freq [GHz]",
                        min_val=23.8,
                        max_val=50.3,
                        step=0.05,
                        default_value=[25.15],
                        on_change=WeatherSatState.preview_5g_freq,
                        on_value_commit=WeatherSatState.set_5g_freq,
                        display_value=WeatherSatState.display_five_g_freq_ghz.to_string() + " GHz",
                        slider_key="wx-5gfreq-" + rk,
                    ),
                    _slider_row(
                        label="EIRP [dBW]",
                        min_val=-8.5,
                        max_val=40.0,
                        step=0.5,
                        default_value=[30.0],
                        on_change=WeatherSatState.preview_5g_eirp,
                        on_value_commit=WeatherSatState.set_5g_eirp,
                        display_value=WeatherSatState.display_five_g_eirp_dbw.to_string() + " dBW",
                        slider_key="wx-5geirp-" + rk,
                    ),
                    _slider_row(
                        label="Emitter density [/km²]",
                        min_val=1.0,
                        max_val=50.0,
                        step=1.0,
                        default_value=[1.0],
                        on_change=WeatherSatState.preview_emitter_density,
                        on_value_commit=WeatherSatState.set_emitter_density,
                        display_value=WeatherSatState.display_emitter_density.to_string() + "/km²",
                        slider_key="wx-dens-" + rk,
                    ),
                    spacing="2",
                ),
                value="fiveg",
            ),
            default_value=["fiveg"],
            type="multiple",
            width="100%",
        ),

        spacing="3",
        width="100%",
        padding="4",
    )
