# RSC-SIM Architecture Summary

## Quick Reference Guide

This document provides a quick overview of the RSC-SIM architecture with links to detailed documentation.

---

## 📚 Documentation Files

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive architecture documentation with detailed module descriptions, physics equations, and design principles

2. **[ARCHITECTURE_MERMAID.md](ARCHITECTURE_MERMAID.md)** - Mermaid diagrams that render beautifully on GitHub/GitLab and can be exported as images

3. **[architecture_diagram.py](architecture_diagram.py)** - Python script to generate a visual architecture diagram (requires matplotlib)

---

## 🏗️ Architecture Layers (Bottom to Top)

### Layer 1: Core Types & I/O
**Files:** `RadioMdl.py`, `radio_types.py`, `radio_io.py`

**Purpose:** Fundamental constants, data structures, and file I/O

**Key Components:**
- Physical constants (k_boltz, speed_c, rad)
- Data classes (Antenna, Instrument, Trajectory, Observation, Constellation, Transmitter)
- File loaders (.cut, .arrow, .csv, .tif)

### Layer 2: Utility Functions
**Files:** `coord_frames.py`, `antenna_pattern.py`

**Purpose:** Coordinate transformations and antenna pattern operations

**Key Functions:**
- Coordinate frame conversions (ground ↔ beam)
- Antenna gain interpolation and calculations
- Radial velocity computations (for Doppler)

### Layer 3: Modeling Modules
**Files:** `astro_mdl.py`, `sat_mdl.py`, `env_mdl.py`

**Purpose:** Physics-based modeling for astronomy, satellites, and environment

**Key Capabilities:**
- **astro_mdl:** Astronomical source fluxes, temperature/power conversions
- **sat_mdl:** Link budget, Doppler effects, transmitter characteristics
- **env_mdl:** Terrain masking, atmospheric effects, DEM analysis

### Layer 4: Observation Modeling
**Files:** `obs_mdl.py`

**Purpose:** Integrate all physics to compute observed system temperature

**Key Functions:**
- `model_observed_temp()` - Standard observation modeling
- `model_observed_temp_vectorized()` - High-performance version
- `model_observed_temp_with_atmospheric_refraction_vectorized()` - Full-featured

### Layer 5: Applications
**Directories:** `educational_tutorials/`, `research_tutorials/`

**Purpose:** User-facing tutorials and examples

**Educational Path:** 01 → 02 → 03 → 04 → 05 → 06 → 07 (progressive complexity)

---

## 🔄 Typical Workflow

```
1. Load Data
   ├─ Antenna pattern (.cut)
   ├─ Trajectories (.arrow)
   └─ DEM data (.tif)

2. Create Types
   ├─ Antenna → Instrument
   ├─ Trajectory → Observation
   └─ Satellites → Constellation

3. Define Models
   ├─ Sky model (Cas A + atmosphere)
   ├─ Link budget (FSPL + Doppler + transmitter)
   └─ Environmental (terrain + atmosphere)

4. Run Simulation
   └─ model_observed_temp_with_atmospheric_refraction_vectorized()

5. Analyze Results
   ├─ Temperature ↔ Power
   ├─ Power Spectral Density
   └─ Plots and comparisons
```

---

## 📦 Module Quick Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| **RadioMdl.py** | Physical constants | `k_boltz`, `speed_c`, `rad` |
| **radio_types.py** | Data structures | `Antenna`, `Instrument`, `Trajectory`, `Observation`, `Constellation`, `Transmitter` |
| **radio_io.py** | File I/O | `power_pattern_from_cut_file()`, file readers/writers |
| **coord_frames.py** | Coordinate transforms | `ground_to_beam_coord()`, `compute_radial_velocity()` |
| **antenna_pattern.py** | Antenna calculations | `interpolate_gain()`, `gain_to_effective_aperture()` |
| **astro_mdl.py** | Astronomical models | `estim_casA_flux()`, `power_to_temperature()` |
| **sat_mdl.py** | Satellite models | `sat_link_budget()`, `lnk_bdgt_with_doppler_correction()` |
| **env_mdl.py** | Environmental effects | `AdvancedEnvironmentalEffects`, terrain/atmosphere modeling |
| **obs_mdl.py** | Observation modeling | `model_observed_temp()` and variants |

---

## 🎯 Key Features by Tutorial

| Tutorial | Features Demonstrated |
|----------|----------------------|
| **01_basic_observation** | Core types, basic observation setup |
| **02_satellite_interference** | Link budget, constellation modeling |
| **03_sky_mapping** | Multiple pointings, beam patterns |
| **04_power_spectral_density** | FFT, frequency analysis |
| **05_doppler_effect** | Frequency shifts, radial velocities |
| **06_transmitter_characteristics** | Polarization, harmonics |
| **07_environment_effects** | Terrain masking, atmospheric refraction, DEM analysis |

---

## 🔬 Physics Models

### Link Budget
```
P_RX = P_TX × G_TX × G_RX × (λ / 4πR)²
```

### System Temperature
```
T_sys = T_RX + (1/4π) × ∫∫ T(θ,φ) × G(θ,φ) dΩ
```

### Doppler Shift
```
f_obs = f_tx × (1 - v_radial / c)
```

### Atmospheric Refraction
```
δ = cot(h + 7.31/(h + 4.4)) / 60  [degrees]
```

---

## 🚀 Performance

| Operation | Non-Vectorized | Vectorized | Speedup |
|-----------|---------------|-----------|---------|
| Link budget (100 sats) | ~1-10 sec | ~0.01-0.1 sec | **10-100x** |
| Observation modeling | ~60 sec | ~1-5 sec | **12-60x** |
| Environmental effects | ~120 sec | ~2-10 sec | **12-60x** |

**Key Optimization:** Use vectorized versions of functions wherever possible!

---

## 📊 Data Formats

| Format | Purpose | Key Fields |
|--------|---------|-----------|
| **.cut** | Antenna patterns | alpha, beta, power |
| **.arrow** | Trajectories | time_stamps, azimuths, elevations, distances |
| **.tif** | DEM data | Elevation grid (GeoTIFF) |
| **.csv** | Generic data | Flexible column structure |

---

## 🛠️ Extension Points

### 1. Custom Sky Model
```python
def my_sky_model(dec, caz, time, freq):
    return T_source + T_atmosphere + T_background
```

### 2. Custom Link Budget
```python
def my_link_budget(dec_tel, caz_tel, instru_tel, 
                   dec_sat, caz_sat, rng_sat, 
                   instru_sat, freq, **kwargs):
    base = sat_link_budget_vectorized(...)
    return base * my_custom_factor
```

### 3. Custom Environmental Effects
```python
class MyEnvironment(AdvancedEnvironmentalEffects):
    def my_custom_effect(self, ...):
        # Your implementation
        pass
```

---

## 📖 Learning Path

### 🟢 Beginner (1-2 hours)
1. Read [README.md](README.md)
2. Run `01_basic_observation.py`
3. Run `02_satellite_interference.py`
4. Understand core concepts

### 🟡 Intermediate (3-5 hours)
5. Run tutorials 03-04
6. Explore [ARCHITECTURE.md](ARCHITECTURE.md)
7. View diagrams in [ARCHITECTURE_MERMAID.md](ARCHITECTURE_MERMAID.md)
8. Modify tutorial parameters

### 🔴 Advanced (5-10 hours)
9. Run tutorials 05-07
10. Study `research_tutorials/`
11. Create custom sky models
12. Generate custom trajectories

### 🔬 Research (Ongoing)
13. Adapt for your observatory
14. Implement custom physics
15. Contribute to the project

---

## 🔗 Quick Links

- **Main README:** [README.md](README.md)
- **Educational Tutorials:** [educational_tutorials/README.md](educational_tutorials/README.md)
- **Research Tutorials:** [research_tutorials/README.md](research_tutorials/README.md)
- **Source Code:** [src/](src/)
- **Example Data:** [research_tutorials/data/](research_tutorials/data/)

---

## 📞 Support

- **Issues:** GitHub issue tracker
- **Email:** dkwon@nd.edu
- **License:** MIT
- **Python:** 3.9+
- **Version:** 1.2.0

---

## 🎓 Citation

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

## 📝 Quick Tips

1. **Always use vectorized functions** for performance
2. **Start with educational tutorials** before research examples
3. **Check data file paths** if you encounter errors
4. **Use virtual environments** for clean dependency management
5. **Refer to ARCHITECTURE.md** for detailed module documentation
6. **View ARCHITECTURE_MERMAID.md on GitHub** for beautiful diagrams
7. **Read tutorial docstrings** for detailed explanations

---

## 🔍 Common Questions

**Q: Which tutorial should I start with?**  
A: Start with `01_basic_observation.py` and work through them sequentially.

**Q: How do I visualize the architecture?**  
A: View [ARCHITECTURE_MERMAID.md](ARCHITECTURE_MERMAID.md) on GitHub or run `architecture_diagram.py`.

**Q: Where are the data files?**  
A: Example data is in `research_tutorials/data/`. Educational tutorials generate demo data.

**Q: How do I add environmental effects?**  
A: See `07_environment_effects.py` for a complete example with DEM data.

**Q: Can I use my own antenna pattern?**  
A: Yes! Use `Antenna.from_file()` with your `.cut` file.

**Q: How do I optimize performance?**  
A: Use vectorized functions (`_vectorized` suffix) and batch operations.

---

**Last Updated:** 2025  
**Maintainer:** SpectrumX Flagship 2  
**License:** MIT

