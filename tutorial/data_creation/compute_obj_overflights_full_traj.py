"""
Compute the trajectories of the objects over the Westford telescope.

Note: change t0 and t1 for the time range of interest

"""

import pyproj
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# convert time to string
def safe_time_str(dt):
    return dt.strftime('%Y-%m-%dT%H_%M_%S.%f')[:-3]  # up to milliseconds


# TELESCOPE POSITION

# Westford coordinates
WESTFORD_X = 1492206.5970
WESTFORD_Y = -4458130.5170
WESTFORD_Z = 4296015.5320
# Observed offset from WGS84 ellipsoid location and Westford STK file
# Not sure which is really right. Need to check with Chet / Diman
WESTFORD_Z_OFFSET = 0.1582435

transformer = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
    )
lon1, lat1, alt1 = transformer.transform(WESTFORD_X, WESTFORD_Y, WESTFORD_Z, radians=False)
print(lat1, lon1, alt1)

WESTFORD_LAT = lat1
WESTFORD_LON = lon1
WESTFORD_ALT = alt1 + WESTFORD_Z_OFFSET

# observer's location
westford = wgs84.latlon(WESTFORD_LAT, WESTFORD_LON, WESTFORD_ALT)


# LOAD OBJECTS

# time scale
ts = load.timescale()

# load Earth
planets = load('traj_files/de421.bsp')
earth = planets['earth']

# load Moon
moon = planets['moon']

# load Sun
sun = planets['sun']

# load Hipparcos catalogue
with load.open(hipparcos.URL, filename='traj_files/hipparcos.dat', reload=False) as f:
    df = hipparcos.load_dataframe(f)

# load Cas A
casA = Star.from_dataframe(df.loc[8148])

# load Cygnus A
cygA = Star(ra_hours=(19, 59, 28.35656837), dec_degrees=(40, 44, 02.0972325))

# load W3(OH)
# w3oh = Star(ra_hours=(2, 27, 4.1), dec_degrees=(61, 52, 22))
w3oh = Star(ra_hours=(2, 27, 3.87), dec_degrees=(61, 52, 24.6), radial_km_per_s=-45)

# load Virgo
vir = Star(ra_hours=(12, 26, 32.1), dec_degrees=(12, 43, 24))


# OBSERVATION

# time of observation
fmt = '%Y-%m-%dT%H:%M:%S.%f'
t0 = ts.utc(2025, 4, 1, 12, 30, 00)
t1 = ts.utc(2025, 4, 1, 13, 30, 00)
# bug fix
dt0 = t0.utc_datetime()
dt1 = t1.utc_datetime()

# time resolution of trajectories
# time_step = [timedelta(milliseconds=1000), timedelta(minutes=30),
#              timedelta(minutes=30), timedelta(minutes=30),
#              timedelta(minutes=30), timedelta(hours=1),
#              timedelta(hours=1), timedelta(milliseconds=1000),
#              timedelta(hours=1)]
time_step = [timedelta(milliseconds=1000)]
# store objects
# objs = [w3oh, moon, sun, casA, cygA, moon, sun, casA, cygA]
objs = [casA]
# name of objects for storage
# str_objs = ['W3(OH)', 'Moon', 'Sun', 'casA', 'cygA', 'Moon', 'Sun', 'casA', 'cygA']
str_objs = ['casA']

# position of Westford on Earth
pos_Wes = earth+westford

# min altitude of object
min_alt = 5  # in degrees

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=True)
for i in range(len(objs)):
    obj = objs[i]
    time_range = pd.date_range(dt0, dt1, freq=time_step[i], tz='UTC')
    temps = []
    alts = []
    azs = []
    diss = []
    for t in time_range:
        pos_obj = pos_Wes.at(ts.from_datetime(t.to_pydatetime())).observe(obj).apparent()
        alt, az, dis = pos_obj.altaz()
        if alt.degrees > min_alt:
            temps.append(t.strftime(fmt)[:-3])
            alts.append(alt.degrees)
            azs.append(az.degrees)
            diss.append(dis.km)
    traj_obj = pd.DataFrame({'time_stamps': temps, 'altitudes': alts, 'azimuths': azs, 'distances': diss})
    # bug fix
    start_str = safe_time_str(dt0)
    end_str = safe_time_str(dt1)
    filename = f"traj_files/{str_objs[i]}_trajectory_Westford_{start_str}_{end_str}.arrow"
    traj_obj.to_feather(filename)

    ax.scatter(np.deg2rad(azs), [90 - a for a in alts], label=str_objs[i])

ax.set_yticks(range(0, 90, 10))
ax.set_yticklabels(map(str, range(90, 0, -10)))
ax.set_theta_zero_location("N")
ax.legend()
plt.show()
