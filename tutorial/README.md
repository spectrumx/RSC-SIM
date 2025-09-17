### Directory Structure

The Python implementation is organized as follows:
- **tuto_radiomdl.ipynb**: Jupyter notebook for python which produces the same results with the Julia notebook (`Julia/test/tuto_mdl_obs_modified.ipynb`)
- **tuto_radiomdl.py**: Python script to run in a command line interface (CLI) that is equivalent to the above notebook
- **tuto_radiomdl_doppler.py**: Enhanced Python script that extends `tuto_radiomdl.py` with Doppler effect analysis and compensation. It includes automatic risk assessment for satellite interference, radial velocity calculations, and physics-based Doppler correction in the frequency domain for more accurate satellite interference predictions
- ***tuto_radiomdl_250401.ipynb and .py***: Testing Jupyter notebook and python script with newly generated Arrow files from the data creation scripts for different time span (see **Data Creation Scripts** section below)
- **Data directory**: `tutorial/data/` - Contains input data files
- **Data creation directory**: `tutorial/data_creation/` - Contains scripts to generate input data files, .arrow, which are for getting trajectories of a star and a satellite with user-specified date and time. Currently, Cas A (Cassiopeia A) and Starlink satellite are used


### Data Files

The `src/python/data/` directory contains input data files of simulations including:
- One **.cut** file: Gain pattern of the MIT Westford antenna generated from TRICA software
- Two **.arrow** files: Trajectory files for astronomical objects (e.g., Cas A) and satellites (e.g., Starlink)


### Data Creation Scripts

The `src/python/data_creation/` directory contains Python scripts that generate Arrow input data files (trajectory files):
- **Stars**: Currently supports Cas A trajectory calculations
- **Satellites**: Currently supports Starlink trajectory calculations

The `src/python/data_creation/traj_files` directory contains two input files for python scripts to generate Arrow files and it also stores generated arrow files:
- **de421.bsp**: positions for planets and their moons for time spans, e.g., https://rhodesmill.org/skyfield/planets.html
- **hipparcos.dat**: Hipparcos catalogue


### Usage

The Python implementation can be used through:
- Direct Python scripts in CLI (e.g., `tuto_radiomdl.py`)
- Jupyter notebooks (e.g., `tuto_radiomdl.ipynb`)
