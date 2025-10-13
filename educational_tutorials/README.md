# RSC-SIM Educational Tutorials

This directory contains a series of focused, educational tutorials for learning radio astronomy observation modeling with the RSC-SIM framework. Each tutorial builds upon the previous ones, providing a progressive learning experience.

<br />


## 📁 Directory Structure

```
educational_tutorials/
├── README.md                          # This file
├── 01_basic_observation.py            # Basic radio astronomy concepts
├── 02_satellite_interference.py       # Satellite modeling
├── 03_sky_mapping.py                  # Sky visualization
├── 04_power_spectral_density.py       # Frequency analysis
├── 05_doppler_effect.py               # Doppler effect analysis and compensation
├── 06_transmitter_characteristics.py  # Transmitter characteristics
├── 07_environment_effects.py          # Environmental effects analysis
├── 08_direct_aggregate_effects.py     # Direct vs aggregate satellite effects
└── shared/                            # Shared utilities
    ├── __init__.py                    # Package initialization
    ├── config.py                      # Configuration parameters
    ├── instrument_setup.py            # Instrument configuration
    ├── sky_models.py                  # Sky temperature models
    └── plotting_utils.py              # Plotting functions
```


## 📚 Overview

### **01: Basic Radio Astronomy Observation**
**File:** `01_basic_observation.py`

**Focus:** Fundamental concepts and basic observation setup

**Related to** `research_tutorials/tuto_radiomdl.py`

**Learning Objectives:**
- Understand radio telescope instrument components
- Learn ON/OFF source observation techniques
- Explore sky temperature models
- Run basic observation simulations
- Analyze signal-to-noise ratios

### **02: Satellite Interference Analysis**
**File:** `02_satellite_interference.py`

**Focus:** Satellite constellation modeling and interference assessment

**Related to** `research_tutorials/tuto_radiomdl.py`

**Learning Objectives:**
- Set up satellite constellation models
- Understand beam avoidance techniques
- Compare interference scenarios
- Analyze mitigation strategies

### **03: Sky Mapping and Visualization**
**File:** `03_sky_mapping.py`

**Focus:** Full sky mapping and spatial analysis

**Related to** `research_tutorials/tuto_radiomdl.py`

**Learning Objectives:**
- Create sky temperature maps
- Visualize satellite positions
- Analyze spatial interference patterns
- Generate comparative sky maps

### **04: Power Spectral Density Analysis**
**File:** `04_power_spectral_density.py`

**Focus:** Frequency domain analysis and spectral characterization

**Related to** `tuto_radiomdl_direct.py/tuto_radiomdl.py`

**Learning Objectives:**
- Understand PSD concepts
- Analyze frequency-dependent interference
- Create spectrograms
- Characterize spectral signatures

### **05: Doppler Effect Analysis and Compensation**
**File:** `05_doppler_effect.py`

**Focus:** Doppler effect analysis and frequency-domain compensation

**Related to** `tuto_radiomdl_direct.py/tuto_radiomdl_doppler.py`

**Learning Objectives:**
- Analyze Doppler shifts in satellite trajectories
- Calculate radial velocities and frequency shifts
- Perform multi-satellite Doppler statistics
- Implement risk-based compensation strategies
- Apply physics-based frequency correction

### **06: Transmitter Characteristics Analysis**
**File:** `06_transmitter_characteristics.py`

**Focus:** Transmitter characteristics and realistic interference modeling

**Related to** `tuto_radiomdl_direct.py/tuto_radiomdl_transmitter.py`

**Learning Objectives:**
- Understand polarization mismatch between satellite transmitters and radio telescopes
- Learn to calculate and visualize polarization loss effects
- Explore harmonic contributions from satellite transmitters
- Implement transmitter modeling with realistic characteristics
- Compare interference predictions with and without transmitter characteristics
- Analyze realistic scenarios (Starlink circular + Westford linear = 3 dB loss)

### **07: Environmental Effects Analysis**
**File:** `07_environment_effects.py`

**Focus:** Environmental effects and realistic propagation modeling

**Related to** `tuto_radiomdl_direct.py/tuto_radiomdl_environment.py`

**Learning Objectives:**
- Understand terrain masking and line-of-sight obstruction effects
- Learn atmospheric refraction modeling and correction techniques
- Explore water vapor absorption and emission in radio astronomy
- Consider limb refraction for space-to-space signal path atmospheric effects
- Implement comprehensive environmental effects in link budget calculations
- Analyze realistic scenarios with terrain and atmospheric effects
- Compare interference predictions with and without environmental effects

### **08: Direct vs Aggregate Satellite Effects Analysis**
**File:** `08_direct_aggregate_effects.py`

**Focus:** Core distinction between direct (one-to-one) and aggregate (many-to-one) satellite effects

**Related to** `tuto_radiomdl_direct.py/tuto_radiomdl_direct.py`

**Learning Objectives:**
- Understand the difference between direct (one-to-one) and aggregate (many-to-one) satellite effects
- Learn to implement single satellite direct effects analysis
- Learn to implement many-to-one aggregate interference calculations
- Visualize direct vs aggregate interference patterns

**Key Concepts:**
- **Direct Effects:** RF interaction of a specific emitter (single satellite) and receiver
- **Aggregate Effects:** Interaction of an ensemble of emitters (multiple satellites) with a given receiver
- **Architecture Shift:** From one-to-one to many-to-one interference calculations
- **Visual Comparison:** Direct visualization of interference enhancement patterns


## 🛠️ Shared Utilities

The `shared/` directory contains reusable components:

### **Configuration (`config.py`)**
- All shared parameters and constants
- File paths and observation settings
- Instrument specifications
- Plotting parameters

### **Instrument Setup (`instrument_setup.py`)**
- Telescope configuration functions
- Satellite transmitter setup
- PSD instrument configuration
- Frequency-dependent models

### **Sky Models (`sky_models.py`)**
- Atmospheric temperature models
- Background radiation models
- Source temperature functions
- Complete sky model creation

### **Plotting Utilities (`plotting_utils.py`)**
- Standard plotting functions
- Polar coordinate visualizations
- Time series plots
- Spectrogram creation

## 🚀 Getting Started

### **Prerequisites**
1. **Python Environment:** Python 3.9+ with required packages
2. **RSC-SIM Installation:** Ensure the main package is installed
3. **Data Files:** Verify tutorial data files are available in `../research_tutorials/data/`

### **Running**
```bash
# Navigate to the educational_tutorials directory
cd educational_tutorials

# Run the first tutorial
python 01_basic_observation.py

# Run with specific Python interpreter if needed
python3 01_basic_observation.py
```

## 🐛 Troubleshooting

### **Common Issues**

**Import Errors:**
```bash
# Ensure you're in the correct directory
cd educational_tutorials

# Check Python path includes src directory
python -c "import sys; print('\\n'.join(sys.path))"
```

**Missing Data Files:**
```bash
# Verify data files exist
ls ../research_tutorials/data/
```

**Plotting Issues:**
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```
