# RSC-SIM

Radio Science Coexistence Simulator RSC-SIM is a sophisticated Python framework designed for modeling radio astronomy observations, with particular emphasis on simulating the impact of satellite constellations (such as Starlink) on radio telescope observations. The framework provides end-to-end simulation capabilities from trajectory generation to power spectral density analysis, enabling researchers and astronomers to assess and mitigate interference from satellite mega-constellations.

<br />

This software grew and evolved out of the RadioMdl codebase, a Julia package created by Dr. Samuel Thé (https://github.com/SJJThe). Modified version of the Julia codes are found at `Julia/` directory.

## Requirements

- **Python**: 3.9 or later
- **Dependencies**: numpy, scipy, pandas, pyarrow, numba

## Modules (at src/ directory)

- `RadioMdl.py`: Core constants and utilities
- `antenna_pattern.py`: Antenna pattern calculations
- `astro_mdl.py`: Astronomical modeling
- `coord_frames.py`: Coordinate frame transformations
- `obs_mdl.py`: Observation modeling
- `radio_io.py`: Radio I/O operations
- `radio_types.py`: Radio data types
- `sat_mdl.py`: Satellite modeling

## Tutorial Resources

### `modular_tutorials/` directory: Recommanded for learning

This directory contains a series of focused, modular tutorials for learning radio astronomy observation modeling with the RSC-SIM framework. Each tutorial builds upon the previous ones, providing a progressive learning experience. Details can be found in the `README.md` inside the `modular_tutorials/`.

### `tutorial/` directory

This contains additional learning resources and examples and serves as a reference implementation for the modeling and simulation of radio astronomy observations. It consists of Jupyter Notebooks and their CLI scripts, and also includes data creation scripts for trajectory files of a star and satellite at the `tutorial/data_creation/` directory. Details can be found at the `README.md` inside the `tutorial/` directory.

## Installation

### Prerequisites

#### 1. Python Installation

**Windows:**
- Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
- During installation, make sure to check "Add Python to PATH"
- Verify installation: `python --version` in Command Prompt

**macOS:**
- Install using Homebrew: `brew install python@3.9`
- Or download from [python.org](https://www.python.org/downloads/)
- Verify installation: `python3 --version` in Terminal

**Linux (Ubuntu/Debian):**
- Install using package manager: `sudo apt update && sudo apt install python3.9 python3.9-venv python3.9-pip`
- Verify installation: `python3.9 --version`

**Linux (CentOS/RHEL/Fedora):**
- Install using package manager: `sudo dnf install python3.9 python3.9-pip` (Fedora) or `sudo yum install python3.9 python3.9-pip` (CentOS/RHEL)
- Verify installation: `python3.9 --version`

#### 2. Visual Studio Code (VS Code) Installation

**Windows:**
- Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
- Run the installer and follow the setup wizard
- Launch VS Code

**macOS:**
- Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
- Drag the downloaded file to Applications folder
- Launch VS Code

**Linux:**
- Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
- Install the .deb package: `sudo dpkg -i code_*.deb`
- Or use snap: `sudo snap install code --classic`

#### 3. Jupyter Extension Installation in VS Code for Running a Jupyter Notebook

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for "Jupyter" extension by Microsoft
4. Click "Install" on the Jupyter extension
5. Restart VS Code if prompted

#### 4. Project Setup in VS Code

1. Open VS Code
2. Go to File → Open Folder
3. Navigate to where you want to clone the repository
4. Open VS Code's integrated terminal (Ctrl+` or Cmd+`)
5. Follow the installation steps below

### Installation Steps

**Note:** Run these commands in VS Code's integrated terminal (Ctrl+` or Cmd+`)

It's recommended to create a virtual environment before installing the package:

```bash
# Clone the repository at a directory
git clone https://github.com/spectrumx/RSC-SIM.git

# Create and activate virtual environment (recommended)
python -m venv rsc-sim-env
# On Windows:
rsc-sim-env\Scripts\activate
# On macOS/Linux:
source rsc-sim-env/bin/activate

# change directory to RSC-SIM and install dependencies (packages)
cd RSC-SIM
pip install -e .

# When you finish working with RSC-SIM, deactivate the virtual environment
deactivate
```

### Running Jupyter Notebooks in VS Code

After installation, you can run the Jupyter notebooks directly in VS Code:

1. Open a `.ipynb` file from the `tutorial/` directories
2. VS Code will automatically detect it as a Jupyter notebook
3. Select your Python interpreter (the one from your virtual environment)
4. Run cells using Shift+Enter or the Run button
5. Interactive plots will be displayed inline in VS Code

### Dependencies

The package automatically installs the following dependencies:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation
- `pyarrow>=6.0.0` - Data serialization
- `numba>=0.56.0` - JIT compilation for performance

## Usage

When working with the source code directly:

```python
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import modules directly
from radio_types import Antenna, Instrument, Observation
from obs_mdl import model_observed_temp
from astro_mdl import estim_casA_flux, power_to_temperature

# Use functions
result = model_observed_temp(observation, sky_model, constellation)
```

## License

MIT License
