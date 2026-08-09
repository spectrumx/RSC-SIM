"""Controls panel for the Radio Astronomy tab."""
import reflex as rx

from reflex_app.state.radio_astro import RadioAstroState


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


def radio_controls() -> rx.Component:
    # Prefix all slider keys with the reset_key so React remounts them on Reset,
    # forcing each slider thumb back to its default_value.
    rk = RadioAstroState.reset_key.to_string()

    return rx.vstack(
        rx.heading("Controls", size="4"),

        rx.button(
            "Reset",
            on_click=RadioAstroState.reset_to_defaults,
            width="100%",
            variant="outline",
        ),

        # --- Time for sky map ---
        rx.accordion.root(
            rx.accordion.item(
                header="Time for Sky Map",
                content=rx.vstack(
                    _slider_row(
                        label="Minutes from obs start",
                        min_val=0.0,
                        max_val=RadioAstroState.obs_total_minutes,
                        step=0.25,
                        default_value=[5.0],
                        on_change=RadioAstroState.preview_time_offset,
                        on_value_commit=RadioAstroState.set_time_offset,
                        display_value=RadioAstroState.display_time_offset_min.to_string() + " min",
                        slider_key="time-" + rk,
                    ),
                    spacing="2",
                ),
                value="time",
            ),
            default_value=["time"],
            type="multiple",
            width="100%",
        ),

        # --- Telescope ---
        rx.accordion.root(
            rx.accordion.item(
                header="Telescope",
                content=rx.vstack(
                    _slider_row(
                        label="Center freq [GHz]",
                        min_val=RadioAstroState.freq_lo_ghz,
                        max_val=RadioAstroState.freq_hi_ghz,
                        step=0.05,
                        default_value=[11.325],
                        on_change=RadioAstroState.preview_center_freq,
                        on_value_commit=RadioAstroState.set_center_freq,
                        display_value=RadioAstroState.display_center_freq_ghz.to_string() + " GHz",
                        slider_key="freq-" + rk,
                    ),
                    _slider_row(
                        label="Bandwidth [kHz]",
                        min_val=0.1,
                        max_val=500.0,
                        step=0.1,
                        default_value=[1.0],
                        on_change=RadioAstroState.preview_bandwidth,
                        on_value_commit=RadioAstroState.set_bandwidth,
                        display_value=RadioAstroState.display_bandwidth_khz.to_string() + " kHz",
                        slider_key="bw-" + rk,
                    ),
                    _slider_row(
                        label="Min elevation [deg]",
                        min_val=0.0,
                        max_val=30.0,
                        step=1.0,
                        default_value=[5.0],
                        on_change=RadioAstroState.preview_min_elevation,
                        on_value_commit=RadioAstroState.set_min_elevation,
                        display_value=RadioAstroState.display_min_elevation.to_string() + "°",
                        slider_key="elev-" + rk,
                    ),
                    spacing="2",
                ),
                value="telescope",
            ),
            type="multiple",
            width="100%",
        ),

        # --- Satellites ---
        rx.accordion.root(
            rx.accordion.item(
                header="Satellites",
                content=rx.vstack(
                    rx.hstack(
                        rx.switch(
                            checked=RadioAstroState.constellation_enabled,
                            on_change=RadioAstroState.toggle_constellation,
                        ),
                        rx.text("Include Starlink constellation", size="2"),
                        align="center",
                        spacing="2",
                    ),
                    _slider_row(
                        label="Beam avoidance [deg]",
                        min_val=0.0,
                        max_val=20.0,
                        step=1.0,
                        default_value=[0.0],
                        on_change=RadioAstroState.preview_beam_avoid,
                        on_value_commit=RadioAstroState.set_beam_avoid,
                        display_value=RadioAstroState.display_beam_avoid_deg.to_string() + "°",
                        slider_key="avoid-" + rk,
                        disabled=~RadioAstroState.constellation_enabled,
                    ),
                    _slider_row(
                        label="# satellites (0 = all)",
                        min_val=0,
                        max_val=RadioAstroState.n_sats_max,
                        step=1,
                        default_value=[0],
                        on_change=RadioAstroState.preview_n_sats,
                        on_value_commit=RadioAstroState.set_n_sats,
                        display_value=RadioAstroState.display_n_sats.to_string(),
                        slider_key="nsats-" + rk,
                        disabled=~RadioAstroState.constellation_enabled,
                    ),
                    rx.text("Direct mode (one satellite)", size="1", weight="bold"),
                    rx.select(
                        RadioAstroState.satellite_options,
                        value=RadioAstroState.direct_sat,
                        on_change=RadioAstroState.set_direct_sat,
                        disabled=~RadioAstroState.constellation_enabled,
                        width="100%",
                    ),
                    spacing="2",
                ),
                value="satellites",
            ),
            default_value=["satellites"],
            type="multiple",
            width="100%",
        ),

        # --- Sky map resolution ---
        rx.accordion.root(
            rx.accordion.item(
                header="Sky Map Resolution",
                content=rx.vstack(
                    _slider_row(
                        label="Az step [deg] (larger = faster)",
                        min_val=5,
                        max_val=30,
                        step=5,
                        default_value=[20],
                        on_change=RadioAstroState.preview_skymap_step,
                        on_value_commit=RadioAstroState.set_skymap_step,
                        display_value=RadioAstroState.display_skymap_step.to_string() + "°",
                        slider_key="step-" + rk,
                    ),
                    rx.button(
                        "Generate Sky Map",
                        on_click=RadioAstroState.request_sky_map,
                        width="100%",
                        size="2",
                        loading=RadioAstroState.sky_map_loading,
                    ),
                    spacing="2",
                ),
                value="skymap",
            ),
            type="multiple",
            width="100%",
        ),

        spacing="3",
        width="100%",
        padding="4",
    )
