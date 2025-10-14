# RSC-SIM Architecture Documentation

## Radio Science Coexistence Simulator - Python Framework

**Version:** 1.2.0  
**Purpose:** Radio astronomy observation modeling with satellite interference analysis

---

## 📐 Architecture Overview

RSC-SIM follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ Educational Tutorials    │  │ Research Tutorials       │   │
│  │ (01-07)                  │  │ (Production Examples)    │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲ ▲
                              │ │
┌─────────────────────────────────────────────────────────────────┐
│                   OBSERVATION MODELING LAYER                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              obs_mdl.py                                  │   │
│  │  • model_observed_temp()                                 │   │
│  │  • model_observed_temp_vectorized()                      │   │
│  │  • model_observed_temp_with_atmospheric_refraction()     │   │
│  │  Integrates: Sky + Constellation + Environmental         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲ ▲ ▲
                              │ │ │
┌─────────────────────────────────────────────────────────────────┐
│                    MODELING MODULES LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │ astro_mdl   │  │  sat_mdl    │  │    env_mdl           │   │
│  │             │  │             │  │                      │   │
│  │ • Cas A     │  │ • Link      │  │ • Terrain masking   │   │
│  │   flux      │  │   budget    │  │ • Atmospheric       │   │
│  │ • Temp/     │  │ • Doppler   │  │   refraction        │   │
│  │   power     │  │ • Transmit  │  │ • Water vapor       │   │
│  │   convert   │  │   effects   │  │ • DEM analysis      │   │
│  │ • ITU       │  │ • Polariz.  │  │ • Line of sight     │   │
│  │   pattern   │  │ • Harmonics │  │                      │   │
│  └─────────────┘  └─────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲ ▲
                              │ │
┌─────────────────────────────────────────────────────────────────┐
│                   UTILITY FUNCTIONS LAYER                        │
│  ┌──────────────────────┐  ┌───────────────────────────────┐   │
│  │  coord_frames.py     │  │   antenna_pattern.py          │   │
│  │  • Coordinate        │  │   • Gain interpolation        │   │
│  │    transforms        │  │   • Pattern mapping           │   │
│  │  • Ground ↔ Beam     │  │   • Effective aperture        │   │
│  │  • Radial velocity   │  │   • Radiated power → gain     │   │
│  └──────────────────────┘  └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  CORE TYPES & I/O LAYER                          │
│  ┌────────────┐  ┌──────────────────────────┐  ┌───────────┐  │
│  │ RadioMdl   │  │     radio_types.py       │  │ radio_io  │  │
│  │            │  │                          │  │           │  │
│  │ Constants: │  │ Data Structures:         │  │ • .cut    │  │
│  │ • k_boltz  │  │ • Antenna                │  │   files   │  │
│  │ • speed_c  │  │ • Instrument             │  │ • .arrow  │  │
│  │ • rad      │  │ • Trajectory             │  │   files   │  │
│  │            │  │ • Observation            │  │ • CSV     │  │
│  │            │  │ • Constellation          │  │           │  │
│  │            │  │ • Transmitter            │  │           │  │
│  └────────────┘  └──────────────────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Descriptions

### Core Layer

#### 1. **RadioMdl.py**
**Purpose:** Fundamental physical constants and conversion factors

**Exports:**
- `k_boltz` = 1.380649e-23 J/K (Boltzmann's constant)
- `speed_c` = 3e8 m/s (speed of light)
- `rad` = π/180 (degree to radian conversion)

**Dependencies:** None

---

#### 2. **radio_types.py**
**Purpose:** Core data structures for radio astronomy modeling

**Classes:**
- **`Antenna`**
  - Stores: gain pattern, interpolator, radiation efficiency
  - Methods: `from_dataframe()`, `from_file()`, `get_gain_values()`
  
- **`Instrument`**
  - Stores: antenna, physical temp, frequency, bandwidth, signal function
  - Methods: `from_scalar()`, `get_center_freq_chans()`
  
- **`Trajectory`**
  - Stores: times, azimuths, elevations, distances
  - Methods: `from_file()`, `get_traj_between()`
  
- **`Observation`**
  - Combines: Trajectory + Instrument + Results array
  - Methods: `from_dates()`, filtering capabilities
  
- **`Constellation`**
  - Stores: satellite DataFrame, transmitter, link budget model
  - Methods: `from_observation()`, `from_file()`
  
- **`Transmitter`**
  - Extends Instrument with: polarization, harmonics
  - Methods: `add_harmonic()`, `get_harmonic_frequencies()`

**Dependencies:** pandas, numpy, scipy, pyarrow

---

#### 3. **radio_io.py**
**Purpose:** Input/output operations for radio astronomy data

**Functions:**
- `power_pattern_from_cut_file()` - Load antenna patterns
- `read_arrow_file()` - Read trajectory data
- `write_arrow_file()` - Write trajectory data

**Supported Formats:**
- `.cut` - Antenna pattern files
- `.arrow` - Apache Arrow trajectory files
- `.csv` - CSV data files

---

### Utility Layer

#### 4. **coord_frames.py**
**Purpose:** Coordinate system transformations

**Key Functions:**
- `ground_to_beam_coord()` - Ground → Beam frame
- `ground_to_beam_coord_vectorized()` - Vectorized version
- `beam_to_ground_coord()` - Beam → Ground frame
- `azimuth_elevation_to_xyz()` - Spherical → Cartesian
- `compute_radial_velocity()` - Doppler velocity calculation

**Coordinate Systems:**
- **Ground Frame:** North-West-Up (antenna reference)
- **Beam Frame:** Antenna pointing direction
- **Spherical:** Azimuth-Elevation-Range

---

#### 5. **antenna_pattern.py**
**Purpose:** Antenna pattern manipulation and calculations

**Key Functions:**
- `map_sphere()` - Map pattern to spherical coordinates
- `interpolate_gain()` - Create gain interpolator (RegularGridInterpolator)
- `radiated_power_to_gain()` - Convert radiated power to gain
- `gain_to_effective_aperture()` - Gain → effective area conversion

**Pattern Representation:**
- Alpha: angle from z-axis (0° = boresight)
- Beta: azimuthal angle (0-360°)
- Gains: linear scale (not dB)

---

### Modeling Layer

#### 6. **astro_mdl.py**
**Purpose:** Astronomical modeling and radio source calculations

**Key Functions:**

**Source Modeling:**
- `estim_casA_flux(freq)` - Cas A flux (Baars et al. 1977 + decay)
- `estim_virgoA_flux(freq)` - Virgo A flux estimation
- `estim_temp(flux, A_eff)` - Flux → temperature conversion

**Power/Temperature Conversions:**
- `power_to_temperature(P, BW)` - Power → temperature (Kelvin)
- `temperature_to_power(T, BW)` - Temperature → power (Watts)
- `temperature_to_flux(T, A_eff)` - Temperature → flux (Jansky)

**Antenna Models:**
- `antenna_mdl_ITU(gain_max, HPBW, ...)` - ITU-recommended pattern

---

#### 7. **sat_mdl.py**
**Purpose:** Satellite interference modeling and link budget calculations

**Core Functions:**

**Basic Link Budget:**
- `sat_link_budget()` - Fundamental link budget (FSPL + antenna gains)
- `sat_link_budget_vectorized()` - High-performance vectorized version

**Doppler Effects:**
- `calculate_doppler_shift(v_radial, freq)` - Frequency shift calculation
- `lnk_bdgt_with_doppler_correction()` - Link budget with Doppler
- `calculate_doppler_from_trajectory()` - Compute radial velocities

**Transmitter Characteristics:**
- `calculate_polarization_mismatch_loss()` - Polarization loss
- `calculate_harmonic_contribution()` - Harmonic interference
- `sat_link_budget_with_polarization()` - Polarization-aware link budget
- `sat_link_budget_with_harmonics()` - Include harmonic components

**Comprehensive Models:**
- `sat_link_budget_comprehensive()` - All transmitter effects
- `link_budget_doppler_transmitter()` - Doppler + transmitter effects

**Environmental Effects:**
- `calculate_comprehensive_environmental_effects()` - Terrain + atmosphere
- `calculate_comprehensive_environmental_effects_vectorized()` - Vectorized

**Link Budget Equation:**
```
P_received = P_transmitted × G_transmitter × G_receiver × (λ / 4πR)²
           × Environmental_Factor × Polarization_Loss
           + Harmonic_Contributions
```

---

#### 8. **env_mdl.py**
**Purpose:** Environmental effects modeling (terrain, atmosphere)

**Main Class: `AdvancedEnvironmentalEffects`**

**Initialization:**
```python
environment = AdvancedEnvironmentalEffects(
    dem_file,           # Digital Elevation Model (GeoTIFF)
    antenna_lat,        # Antenna latitude (degrees)
    antenna_lon,        # Antenna longitude (degrees)
    antenna_elevation,  # Antenna height above ground (m)
    temperature,        # Surface temperature (K)
    pressure,           # Surface pressure (Pa)
    humidity            # Relative humidity (%)
)
```

**Key Methods:**

**Terrain Masking:**
- `load_dem()` - Load and process DEM data
- `check_line_of_sight(alt, az, range)` - Ray tracing through terrain
- `apply_terrain_masking(alt, az, range)` - Masking factor calculation
- `apply_terrain_masking_vectorized()` - Vectorized version

**Atmospheric Effects:**
- `calculate_atmospheric_refraction(elevation)` - Bennett's formula + enhanced
- `calculate_water_vapor_absorption(freq, elevation)` - H₂O absorption
- `calculate_integrated_atmospheric_effects()` - Complete atmosphere model
- `calculate_atmospheric_profile(height)` - T, P, ρ_H₂O profiles

**Antenna Constraints:**
- `check_antenna_limitations(elevation)` - Mechanical pointing limits
- `check_elevation_masking(elevation)` - Minimum elevation threshold

**Space Effects:**
- `apply_limb_refraction(grazing_angle)` - Atmosphere limb effects

**Environmental Factors:**
- Terrain blocking: Binary (visible/not visible)
- Atmospheric loss: Frequency-dependent attenuation
- Refraction: Pointing correction (typically 0.01-0.5°)
- Water vapor: Absorption + emission (significant >10 GHz)

---

#### 9. **obs_mdl.py**
**Purpose:** Observation modeling - integrates all physics

**Core Functions:**

**Standard Observation:**
- `model_observed_temp(observation, sky_mdl, constellation)`
  - Computes: T_sys = T_receiver + T_antenna + T_sky + T_interference
  - Vectorized across time, pointings, frequencies
  - Optional beam avoidance

**With Atmospheric Refraction:**
- `model_observed_temp_with_atmospheric_refraction_vectorized()`
  - Adds: Pointing corrections for atmospheric refraction
  - Category 1 effects: Link budget modifications
  - Category 2 effects: Telescope pointing corrections

**Observation Equation:**
```
T_observed = T_receiver + (1/4π) × (T_sky × G_max + Σ T_sat × G_tel × G_sat × FSPL⁻¹)
           × Environmental_Factors × (1 - η_rad) × T_phy
```

**Vectorization Strategy:**
- Time loop (T timesteps)
- Vectorize over satellites (S satellites)
- Vectorize over frequencies (F channels)
- Vectorize over pointings (P sky positions)
- Result shape: (T, P, F)

---

## 🔄 Data Flow Pipeline

### Typical Workflow

```
1. SETUP PHASE
   ├─ Load DEM data (env_mdl)
   ├─ Load antenna pattern (.cut file → Antenna)
   ├─ Create receiver (Antenna + parameters → Instrument)
   └─ Load trajectories (.arrow files → Trajectory)

2. OBSERVATION CREATION
   ├─ Define time window (start, stop)
   ├─ Apply filters (elevation > 5°, terrain masking)
   └─ Create Observation object (Trajectory + Instrument)

3. CONSTELLATION SETUP
   ├─ Load satellite data (.arrow files)
   ├─ Create transmitter (Instrument + polarization + harmonics)
   ├─ Define link budget model (with Doppler, transmitter, environment)
   └─ Create Constellation (satellites + transmitter + link budget)

4. MODELING
   ├─ Define sky model (Cas A + atmosphere + CMB + galactic)
   ├─ Choose observation function:
   │  ├─ model_observed_temp() [basic]
   │  └─ model_observed_temp_with_atmospheric_refraction_vectorized() [full]
   └─ Run simulation (returns temperature time series)

5. ANALYSIS
   ├─ Convert temperature ↔ power
   ├─ Compute power spectral density
   ├─ Plot results
   └─ Compare with/without environmental effects
```

---

## 📊 Key Physics Models

### 1. **Free Space Path Loss (FSPL)**
```
FSPL = (λ / 4πR)² = (c / 4πfR)²
```
- λ: wavelength
- R: distance
- f: frequency
- c: speed of light

### 2. **Antenna Temperature**
```
T_A = (1/4π) × ∫ T(θ,φ) × G(θ,φ) dΩ
```
- T(θ,φ): sky brightness temperature
- G(θ,φ): antenna gain pattern
- Integration over 4π steradians

### 3. **System Temperature**
```
T_sys = T_receiver + T_antenna + T_sky + T_interference
```

### 4. **Doppler Shift**
```
f_observed = f_transmitted × (1 - v_radial/c)
```
- v_radial: radial velocity (positive = moving away)

### 5. **Atmospheric Refraction (Bennett's Formula)**
```
δ = cot(h + 7.31/(h + 4.4)) / 60
```
- δ: refraction correction (degrees)
- h: apparent elevation (degrees)
- Valid for h > 15°

### 6. **Water Vapor Absorption**
```
L_wv = α_wv × sec(z) × ∫ ρ_H₂O(h) dh
```
- α_wv: absorption coefficient (frequency-dependent)
- z: zenith angle
- ρ_H₂O(h): water vapor density profile

---

## 🎯 Design Principles

### 1. **Modularity**
- Each module has single responsibility
- Clear interfaces between layers
- Minimal circular dependencies

### 2. **Vectorization**
- NumPy arrays for batch operations
- Numba JIT compilation for performance
- Vectorized versions of core functions

### 3. **Flexibility**
- Custom link budget functions
- Pluggable sky models
- Configurable environmental effects

### 4. **Realism**
- Physics-based models (FSPL, Doppler, refraction)
- Real data support (.arrow, .cut, .tif files)
- Production-ready for research

### 5. **Progressive Complexity**
- Educational tutorials: Simple → Complex
- Research tutorials: Full-featured examples
- Fallback to simpler models when needed

---

## 🔧 Performance Optimizations

### Vectorization Strategy
```python
# ❌ Slow: Loop over satellites
for sat in satellites:
    result += compute_interference(sat)

# ✅ Fast: Vectorized computation
result = compute_interference_vectorized(satellites)  # 10-100x faster
```

### Key Optimizations
1. **Pre-computation**: Satellite data cached per timestep
2. **Batch operations**: Process all satellites simultaneously
3. **Numba acceleration**: JIT compilation for critical functions
4. **Memory efficiency**: In-place operations where possible
5. **Lazy evaluation**: Load data only when needed

### Performance Metrics
- Educational demo: ~1-10 seconds (small dataset)
- Research simulation: ~30-120 seconds (10,000+ satellite positions)
- Vectorized vs. non-vectorized: **10-100x speedup**

---

## 📚 Tutorial Organization

### Educational Tutorials (Progressive Learning)

**Directory:** `educational_tutorials/`

1. **01_basic_observation.py**
   - Goal: Learn core concepts
   - Topics: Antenna, Instrument, Observation
   - Data: Synthetic trajectories

2. **02_satellite_interference.py**
   - Goal: Add satellite interference
   - Topics: Constellation, link budget
   - Data: Demo satellites

3. **03_sky_mapping.py**
   - Goal: Multiple sky pointings
   - Topics: Sky grids, beam patterns
   - Data: Grid observations

4. **04_power_spectral_density.py**
   - Goal: Frequency analysis
   - Topics: FFT, PSD computation
   - Data: Time series analysis

5. **05_doppler_effect.py**
   - Goal: Doppler corrections
   - Topics: Radial velocity, frequency shifts
   - Data: Moving satellites

6. **06_transmitter_characteristics.py**
   - Goal: Enhanced transmitter models
   - Topics: Polarization, harmonics
   - Data: Transmitter parameters

7. **07_environment_effects.py**
   - Goal: Environmental realism
   - Topics: Terrain, atmosphere, DEM
   - Data: Real DEM + trajectories

### Research Tutorials (Production Examples)

**Directory:** `research_tutorials/`

- **tuto_radiomdl.py** - Complete workflow
- **tuto_radiomdl_doppler.py** - Doppler-focused
- **tuto_radiomdl_transmitter.py** - Transmitter-focused
- **tuto_radiomdl_environment.py** - Environment-focused
- **data_creation/** - Trajectory generation scripts

---

## 🗂️ File Format Specifications

### Antenna Pattern (.cut file)
```
Format: Space-separated values
Columns: alpha beta power
Units: degrees, degrees, linear
Range: alpha [0, 180], beta [0, 360]
```

### Trajectory (.arrow file)
```
Format: Apache Arrow IPC
Required columns: time_stamps, azimuths, elevations, distances
Units: datetime, degrees, degrees, meters
```

### DEM Data (.tif file)
```
Format: GeoTIFF
Coordinate system: WGS84 or UTM
Elevation units: meters
Resolution: Typically 1m-30m
```

---

## 🔗 Dependencies

### Core Dependencies
- **numpy** ≥ 1.21.0 - Numerical computing
- **scipy** ≥ 1.5.0 - Scientific computing
- **pandas** ≥ 1.3.0 - Data manipulation
- **pyarrow** ≥ 6.0.0 - Arrow file format

### Performance
- **numba** ≥ 0.56.0 - JIT compilation

### Visualization
- **matplotlib** ≥ 3.5.0 - Plotting

### Astronomy
- **skyfield** ≥ 1.40 - Ephemeris calculations

### Geospatial
- **pyproj** ≥ 3.0.0 - Coordinate transformations
- **rasterio** ≥ 1.3.0 - GeoTIFF/DEM handling

### Satellite
- **sgp4** ≥ 2.0.0 - Orbit propagation

---

## 🎓 Learning Path

### Beginner
1. Run **01_basic_observation.py** - understand core types
2. Run **02_satellite_interference.py** - add satellites
3. Modify parameters, observe changes

### Intermediate
4. Run **03_sky_mapping.py** - multiple pointings
5. Run **04_power_spectral_density.py** - frequency analysis
6. Explore custom sky models

### Advanced
7. Run **05_doppler_effect.py** - moving satellites
8. Run **06_transmitter_characteristics.py** - realistic transmitters
9. Run **07_environment_effects.py** - full environmental modeling

### Research
10. Study **research_tutorials/** - production workflows
11. Generate custom trajectories with **data_creation/** scripts
12. Adapt for your observatory and frequency bands

---

## 🚀 Extension Points

### Custom Sky Models
```python
def custom_sky_model(dec, caz, time, freq):
    T_source = ...  # Your source model
    T_atmosphere = ...  # Your atmosphere model
    T_background = ...  # Your background model
    return T_source + T_atmosphere + T_background
```

### Custom Link Budget
```python
def custom_link_budget(dec_tel, caz_tel, instru_tel, 
                       dec_sat, caz_sat, rng_sat, 
                       instru_sat, freq, **kwargs):
    # Your custom physics
    base = sat_link_budget_vectorized(...)
    custom_factor = ...  # Your modifications
    return base * custom_factor
```

### Custom Environmental Effects
```python
class CustomEnvironment(AdvancedEnvironmentalEffects):
    def calculate_custom_effect(self, ...):
        # Your environmental model
        pass
```

---

## 📝 Citation

If you use RSC-SIM in your research, please cite:

```bibtex
@software{rsc_sim,
  title={RSC-SIM: Radio Science Coexistence Simulator},
  author={SpectrumX Flagship 2},
  year={2025},
  version={1.2.0},
  url={https://github.com/spectrumx/RSC-SIM}
}
```

---

## 📧 Contact & Support

- **Issues:** GitHub issue tracker
- **Email:** dkwon@nd.edu
- **License:** MIT License
- **Python:** 3.9+

---

## 🔄 Version History

### v1.2.0 (Current)
- Environmental effects module (env_mdl.py)
- Vectorized observation modeling
- Educational tutorials (01-07)
- Comprehensive documentation

### v1.1.0
- Transmitter characteristics (polarization, harmonics)
- Doppler effect corrections
- Research tutorials

### v1.0.0
- Initial release
- Basic link budget calculations
- Core data types

