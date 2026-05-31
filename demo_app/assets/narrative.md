# Booth narratives (60-second scripts)

These are the suggested talk tracks for each tab. Read them top-to-bottom while moving the matching sliders.

---

## Tab 1: Starlink vs. radio telescope (Looking-Up)

> "This is RSC-SIM simulating the Westford radio telescope tracking Cas A in 2025.
> The polar plot is the sky as the telescope sees it, with the antenna gain pattern
> on the floor and satellites lit up where they happen to be at this moment.
> Power is in **dBW** -- the same units the rest of RSC-SIM's Looking-Up tutorials use.
>
> Right now we have **no satellites** -- a clean observation, the time series
> shows just sky + source noise floor power.
>
> Now I'll **enable the Starlink constellation**. Watch the time series spike --
> every spike is a Starlink crossing the main beam. That's a real RFI event
> showing up as tens of dB of extra received power.
>
> Now I'll turn on **beam avoidance** -- a 10-degree exclusion around each
> satellite's main lobe. The spikes drop dramatically. That's the kind of
> mitigation strategy operators are negotiating with constellation owners today.
>
> If you want to see a single satellite's contribution, switch to **direct mode**
> and pick a name -- this is the building block for forensic analysis of
> interference events."

---

## Tab 2: Weather satellite single FOV (Looking-Down)

> "Now we flip the geometry: a weather satellite (Suomi-NPP / JPSS) looking
> **down** at Earth. Its passive microwave radiometer is trying to measure
> atmospheric brightness temperature, but **Starlink back-lobes** and **5G mmWave**
> towers leak power upward and contaminate the measurement.
>
> Both **K-Band (23.8 GHz)** and **V-Band (50.3 GHz)** channels are shown at once —
> water vapor and oxygen channels used in numerical weather prediction. The chart
> is received **RFI power in dBW**, not brightness temperature: Starlink dashed,
> 5G dotted, for each band.
>
> Drag **time** — the vertical marker moves along the overpass without recomputing.
> The headline tiles update for that instant: visible Starlinks, fundamentals, and
> K-/V-Bands RFIs. The table above the chart is the **peak** RFIs over the whole pass.
>
> Now bump **5G emitter density** or slide **5G fundamental frequency** toward
> 23.8 or 50.3 GHz — watch harmonics land in K-band or V-band. Starlink sliders
> do the same from the other side. That's the coexistence question NWP operators
> care about: terrestrial and constellation RFI in the same channels ATMS uses."

---

## Tips

- Keep the time scrubber moving. A static plot reads as broken; a moving plot reads as live.
- All controls live in the **Controls** column on the left side of each tab; switching tabs swaps the entire control set.
- "Reset to demo defaults" before each new visitor.
