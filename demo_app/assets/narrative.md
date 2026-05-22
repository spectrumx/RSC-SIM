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

## Tab 2: Weather satellite single FOV

> "Now we flip the geometry: a weather satellite (Suomi-NPP / JPSS) looking
> **down** at Earth. Its passive microwave radiometer is trying to measure
> atmospheric brightness temperature, but Starlink **back-lobes** leak power
> upward and contaminate the measurement.
>
> Pick the **K-Band channel at 23.8 GHz** -- water vapor channel, used in
> numerical weather prediction. The bar chart on the right shows the Tb budget:
> Earth, sky, system noise, plus a Starlink contribution.
>
> Now slide to the **V-Band 50.3 GHz** channel -- atmospheric oxygen, used to
> retrieve temperature profiles. Different harmonics of the Starlink fundamental
> end up in different channels, so the Starlink contribution shifts.
>
> If we have the 5G data and gateway file staged, you can also drop a Starlink
> gateway anywhere on the map and watch it light up the FOV when the satellite
> flies over."

---

## Tab 3: Pre-baked gallery

> "Some of the analyses are too heavy for live demo -- like the full
> numerical-weather-prediction pipeline that processes a 12-hour ATMS scan.
> We've baked those results out and you can flip through them here.
>
> On the left, the **native ATMS brightness temperature**. On the right, the
> same scan **after RFI is added**. The difference is what an operational NWP
> assimilation system would see if mitigation isn't applied.
>
> The waterfall in the lower panel is a **Doppler signature** -- one Starlink
> sweeping across our radio receiver band as it passes overhead. The streak is
> the carrier shifting in frequency, exactly the kind of feature that confuses
> automatic RFI flaggers."

---

## Tips

- Keep the time scrubber moving. A static plot reads as broken; a moving plot reads as live.
- All controls live in the **Controls** column on the left side of each tab; switching tabs swaps the entire control set.
- "Reset to demo defaults" before each new visitor.
- If the laptop hiccups, jump to the **Gallery** tab; the images render instantly.
