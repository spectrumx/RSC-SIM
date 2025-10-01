"""
Environmental effects modeling for radio astronomy observations

This module provides comprehensive environmental effects modeling including:
- Terrain masking effects using Digital Elevation Models (DEM)
- Advanced atmospheric refraction models
- Water vapor modeling for high-frequency simulations
- Integrated atmospheric-terrain interactions
"""

import numpy as np
import rasterio
from pyproj import Proj, transform as pyproj_transform


class AdvancedEnvironmentalEffects:
    """
    Comprehensive class to handle advanced propagation and environmental effects including:
    - Terrain masking effects using Digital Elevation Models (DEM)
    - Advanced atmospheric refraction models
    - Water vapor modeling for high-frequency simulations
    - Integrated atmospheric-terrain interactions
    """

    def __init__(self, dem_file, antenna_lat, antenna_lon, antenna_elevation=0,
                 temperature=288.15, pressure=101325, humidity=50.0):
        """
        Initialize environmental effects modeling with DEM data and atmospheric conditions

        Args:
            dem_file: Path to DEM GeoTIFF file
            antenna_lat: Antenna latitude in degrees
            antenna_lon: Antenna longitude in degrees
            antenna_elevation: Antenna elevation above ground in meters
            temperature: Surface temperature in Kelvin (default: 288.15 K = 15°C)
            pressure: Surface pressure in Pa (default: 101325 Pa = 1 atm)
            humidity: Relative humidity in % (default: 50%)
        """
        self.dem_file = dem_file
        self.antenna_lat = antenna_lat
        self.antenna_lon = antenna_lon
        self.antenna_elevation = antenna_elevation

        # Atmospheric conditions
        self.temperature = temperature  # K
        self.pressure = pressure  # Pa
        self.humidity = humidity  # %

        # Load DEM data
        self.dem_data = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_bounds = None
        self.load_dem()

        # Environmental effects parameters
        self.min_elevation_angle = 5.0  # degrees - minimum elevation for observations
        self.refraction_coefficient = 0.13  # Standard atmospheric refraction coefficient
        self.antenna_mechanical_limit = 5.0  # degrees - mechanical pointing limit

        # Water vapor parameters
        self.water_vapor_scale_height = 2000.0  # meters - typical scale height for water vapor
        self.water_vapor_absorption_coeff = 0.1  # dB/km at 22 GHz (simplified)

        # Atmospheric profile parameters
        self.tropopause_height = 11000.0  # meters
        self.atmospheric_scale_height = 8000.0  # meters

    def load_dem(self):
        """Load DEM data from GeoTIFF file"""
        try:
            with rasterio.open(self.dem_file) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_bounds = src.bounds

                print("DEM loaded successfully:")
                print(f"  Shape: {self.dem_data.shape}")
                print(f"  CRS: {self.dem_crs}")
                print(f"  Bounds: {self.dem_bounds}")
                print(f"  Elevation range: {np.nanmin(self.dem_data):.1f} to {np.nanmax(self.dem_data):.1f} m")

        except Exception as e:
            print(f"Error loading DEM: {e}")
            self.dem_data = None

    def latlon_to_dem_coords(self, lat, lon):
        """
        Convert latitude/longitude to DEM pixel coordinates

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            row, col: DEM pixel coordinates
        """
        if self.dem_data is None:
            return None, None

        try:
            # Transform from lat/lon to DEM CRS
            # Create projections
            wgs84 = Proj(init='epsg:4326')
            dem_proj = Proj(self.dem_crs)

            # Transform coordinates
            x, y = pyproj_transform(wgs84, dem_proj, lon, lat)

            # Convert to pixel coordinates
            col, row = rasterio.transform.rowcol(self.dem_transform, x, y)

            # Check bounds
            if 0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]:
                return row, col
            else:
                return None, None

        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None, None

    def get_terrain_elevation(self, lat, lon):
        """
        Get terrain elevation at given latitude/longitude

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            elevation: Terrain elevation in meters, or None if outside DEM bounds
        """
        row, col = self.latlon_to_dem_coords(lat, lon)
        if row is not None and col is not None:
            return self.dem_data[row, col]
        return None

    def calculate_atmospheric_profile(self, height):
        """
        Calculate atmospheric parameters at given height

        Args:
            height: Height above sea level in meters

        Returns:
            temperature, pressure, water_vapor_density: Atmospheric parameters
        """
        # Standard atmospheric model with terrain influence
        if height <= self.tropopause_height:
            # Troposphere: linear temperature decrease
            temperature = self.temperature - 6.5e-3 * height  # K/km lapse rate
            pressure = self.pressure * np.exp(-height / self.atmospheric_scale_height)
        else:
            # Stratosphere: constant temperature
            temperature = self.temperature - 6.5e-3 * self.tropopause_height
            pressure = (self.pressure * np.exp(-self.tropopause_height / self.atmospheric_scale_height) *
                        np.exp(-(height - self.tropopause_height) / (2 * self.atmospheric_scale_height)))

        # Water vapor density (exponential decrease with height)
        # Influenced by terrain - higher terrain can trap moisture
        terrain_factor = 1.0 + 0.1 * (self.antenna_elevation / 1000.0)  # Terrain influence
        water_vapor_density = ((self.humidity / 100.0) * 0.01 * terrain_factor *
                               np.exp(-height / self.water_vapor_scale_height))  # kg/m³

        return temperature, pressure, water_vapor_density

    def apply_advanced_atmospheric_refraction(self, elevation_angle, frequency=11e9):
        """
        Apply advanced atmospheric refraction correction considering atmospheric profile

        Args:
            elevation_angle: True elevation angle in degrees
            frequency: Observation frequency in Hz

        Returns:
            apparent_elevation: Apparent elevation angle after refraction
            atmospheric_delay: Atmospheric delay in meters
        """
        if elevation_angle < 0:
            return elevation_angle, 0.0  # No refraction for negative angles

        # Enhanced refraction model considering atmospheric profile
        # Sample atmosphere along the ray path
        max_height = 50000  # meters
        num_samples = 50
        heights = np.linspace(0, max_height, num_samples)

        total_refraction = 0.0
        total_delay = 0.0

        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i + 1]
            dh = h2 - h1

            # Calculate atmospheric parameters at this height
            T1, P1, WV1 = self.calculate_atmospheric_profile(h1)
            T2, P2, WV2 = self.calculate_atmospheric_profile(h2)

            # Average values for this layer
            T_avg = (T1 + T2) / 2
            P_avg = (P1 + P2) / 2
            WV_avg = (WV1 + WV2) / 2

            # Refractive index (simplified model)
            n_dry = 1 + 77.6e-6 * P_avg / T_avg
            n_wet = 1 + 72.0e-6 * WV_avg * T_avg / P_avg
            n_total = n_dry + n_wet

            # Refraction contribution from this layer
            layer_refraction = (n_total - 1) * dh / (6371000 + h1) * 180 / np.pi
            total_refraction += layer_refraction

            # Atmospheric delay contribution
            layer_delay = (n_total - 1) * dh / np.sin(np.radians(elevation_angle))
            total_delay += layer_delay

        apparent_elevation = elevation_angle + total_refraction
        return apparent_elevation, total_delay

    def apply_atmospheric_refraction(self, elevation_angle):
        """
        Apply atmospheric refraction correction for low elevation angles (legacy method)

        Args:
            elevation_angle: True elevation angle in degrees

        Returns:
            apparent_elevation: Apparent elevation angle after refraction
        """
        if elevation_angle < 0:
            return elevation_angle  # No refraction for negative angles

        # Simple refraction model (Bennett's formula)
        # More sophisticated models would consider temperature, pressure, humidity
        refraction_correction = self.refraction_coefficient / np.tan(np.radians(elevation_angle))

        apparent_elevation = elevation_angle + refraction_correction
        return apparent_elevation

    def calculate_water_vapor_effects(self, elevation_angle, frequency, path_length=None):
        """
        Calculate water vapor absorption and emission effects

        Args:
            elevation_angle: Elevation angle in degrees
            frequency: Observation frequency in Hz
            path_length: Path length through atmosphere in meters (optional)

        Returns:
            absorption_db: Water vapor absorption in dB
            emission_temperature: Water vapor emission temperature in K
        """
        if elevation_angle <= 0:
            return 0.0, 0.0

        # Calculate path length through atmosphere if not provided
        if path_length is None:
            # Simplified path length calculation
            atmosphere_height = 50000  # meters
            path_length = atmosphere_height / np.sin(np.radians(elevation_angle))

        # Sample atmosphere along the path
        num_samples = 20
        heights = np.linspace(0, atmosphere_height, num_samples)

        total_absorption = 0.0
        total_emission = 0.0

        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i + 1]
            dh = h2 - h1

            # Calculate atmospheric parameters
            T, P, WV = self.calculate_atmospheric_profile((h1 + h2) / 2)

            # Water vapor absorption coefficient (simplified model)
            # Based on Liebe's model for frequencies around 22 GHz
            if frequency > 10e9:  # High frequency regime
                # Simplified absorption coefficient
                alpha_wv = self.water_vapor_absorption_coeff * (WV / 0.01) * \
                          (frequency / 22e9) ** 2  # dB/km
            else:
                alpha_wv = 0.01 * (WV / 0.01)  # Lower absorption at lower frequencies

            # Absorption contribution
            layer_absorption = alpha_wv * dh / 1000  # Convert to dB
            total_absorption += layer_absorption

            # Emission contribution (simplified)
            # Water vapor emits thermal radiation
            layer_emission = T * (1 - np.exp(-alpha_wv * dh / 1000 / 4.34))  # K
            total_emission += layer_emission

        return total_absorption, total_emission

    def calculate_integrated_atmospheric_effects(self, elevation_angle, frequency):
        """
        Calculate integrated atmospheric effects including refraction, water vapor, and terrain influence

        Args:
            elevation_angle: True elevation angle in degrees
            frequency: Observation frequency in Hz

        Returns:
            apparent_elevation: Apparent elevation after refraction
            atmospheric_delay: Atmospheric delay in meters
            water_vapor_absorption: Water vapor absorption in dB
            water_vapor_emission: Water vapor emission temperature in K
            total_atmospheric_loss: Total atmospheric loss in dB
        """
        # Advanced refraction calculation
        apparent_elevation, atmospheric_delay = self.apply_advanced_atmospheric_refraction(
            elevation_angle, frequency)

        # Water vapor effects
        water_vapor_absorption, water_vapor_emission = self.calculate_water_vapor_effects(
            elevation_angle, frequency)

        # Terrain influence on atmospheric effects
        # Higher terrain can affect local atmospheric conditions
        terrain_factor = 1.0 + 0.05 * (self.antenna_elevation / 1000.0)

        # Total atmospheric loss (absorption + scattering)
        total_atmospheric_loss = water_vapor_absorption * terrain_factor

        return (apparent_elevation, atmospheric_delay, water_vapor_absorption,
                water_vapor_emission, total_atmospheric_loss)

    def apply_limb_refraction(self, grazing_angle):
        """
        Apply limb refraction effects for space-to-space interactions

        Args:
            grazing_angle: Grazing angle at Earth's limb in degrees

        Returns:
            refraction_effect: Refraction effect in degrees
            signal_bending: Signal bending angle in degrees
        """
        if grazing_angle <= 0:
            return 0.0, 0.0

        # Simplified limb refraction model
        # More sophisticated models would consider atmospheric density profile
        limb_refraction = 0.5 / np.tan(np.radians(grazing_angle))  # degrees
        signal_bending = limb_refraction * 0.1  # simplified relationship

        return limb_refraction, signal_bending

    def check_antenna_limitations(self, elevation_angle):
        """
        Check if elevation angle meets antenna pointing limitations

        Args:
            elevation_angle: Elevation angle in degrees

        Returns:
            is_accessible: Boolean indicating if antenna can point to this elevation
        """
        # Check mechanical limits
        if elevation_angle < self.antenna_mechanical_limit:
            return False

        # Check for additional limitations (e.g., RFI, atmospheric effects)
        # This is a simplified model - real implementations would be more complex
        # Reduced threshold to match the basic elevation filter (5.0 degrees)
        if elevation_angle < 5.0:  # Increased noise at very low elevations
            return False

        return True

    def check_elevation_masking(self, elevation_angle):
        """
        Check if elevation angle is above minimum threshold

        Args:
            elevation_angle: Elevation angle in degrees

        Returns:
            is_visible: Boolean indicating if elevation is sufficient
        """
        # Apply refraction correction
        apparent_elevation = self.apply_atmospheric_refraction(elevation_angle)

        # Check if above minimum elevation angle
        return apparent_elevation >= self.min_elevation_angle

    def check_line_of_sight(self, sat_alt, sat_az, sat_range, frequency=11e9, num_points=10):
        """
        Check if line of sight to satellite is blocked by terrain with integrated atmospheric effects

        Args:
            sat_alt: Satellite altitude in degrees
            sat_az: Satellite azimuth in degrees
            sat_range: Range to satellite in meters
            frequency: Observation frequency in Hz
            num_points: Number of points to sample along the line of sight

        Returns:
            is_visible: Boolean indicating if satellite is visible
            blocking_elevation: Elevation of terrain blocking the view (if any)
            atmospheric_effects: Dictionary with atmospheric effect details
        """
        if self.dem_data is None:
            return True, 0, {}  # No terrain data, assume visible

        # Calculate elevation angle
        elevation_angle = sat_alt  # Simplified - assuming sat_alt is elevation angle

        # Calculate integrated atmospheric effects
        (apparent_elevation, atmospheric_delay, water_vapor_absorption,
         water_vapor_emission, total_atmospheric_loss) = self.calculate_integrated_atmospheric_effects(
            elevation_angle, frequency)

        # TEMPORARY FIX: If atmospheric refraction is making elevation too low, use true elevation
        if apparent_elevation < elevation_angle * 0.5:  # If refraction reduces elevation by more than 50%
            apparent_elevation = elevation_angle

        # Check if below minimum elevation angle (using apparent elevation)
        if apparent_elevation < self.min_elevation_angle:
            return False, 0, {
                'apparent_elevation': apparent_elevation,
                'atmospheric_delay': atmospheric_delay,
                'water_vapor_absorption': water_vapor_absorption,
                'water_vapor_emission': water_vapor_emission,
                'total_atmospheric_loss': total_atmospheric_loss
            }

        # Ray tracing through DEM with atmospheric effects: sample points along line of sight
        # and check if terrain elevation blocks the satellite view

        # Calculate actual satellite altitude from horizontal range and elevation angle
        # Using trigonometry: altitude = horizontal_range * tan(elevation_angle)
        # This gives us the satellite height above the antenna elevation
        actual_sat_altitude = sat_range * np.tan(np.radians(elevation_angle))

        # Sanity check: LEO satellites should be 200-2000 km above Earth's surface
        # If calculated altitude is unreasonable, use a fallback estimate
        if actual_sat_altitude < 200000 or actual_sat_altitude > 2000000:  # 200-2000 km
            # Fallback: use typical LEO altitude (500 km above antenna)
            actual_sat_altitude = 500000

        # OPTIMIZATION: Skip terrain ray tracing for high elevation satellites
        # Satellites above 30° are very unlikely to be blocked by terrain
        if elevation_angle > 30.0:
            return True, 0, {
                'apparent_elevation': apparent_elevation,
                'atmospheric_delay': atmospheric_delay,
                'water_vapor_absorption': water_vapor_absorption,
                'water_vapor_emission': water_vapor_emission,
                'total_atmospheric_loss': total_atmospheric_loss
            }

        # For terrain ray tracing, we only need to check the first few kilometers
        # The satellite is high enough that terrain blocking only occurs in the immediate vicinity

        # Sample points at different distances along the line of sight
        # Limit sampling to stay within DEM bounds (DEM covers ~15 km x 15 km area)
        # For terrain masking, we only need to check the first few kilometers
        max_sample_distance = 10000  # 10 km - should be within DEM bounds
        distances = np.linspace(100, max_sample_distance, num_points)  # Start from 100m

        for i, dist in enumerate(distances):
            # Calculate position along line of sight using proper spherical geometry
            # Convert azimuth and elevation to geographic coordinates

            # Calculate the horizontal distance to the sample point
            horizontal_dist = dist * np.cos(np.radians(elevation_angle))

            # Convert to geographic coordinates (simplified spherical projection)
            # This is a simplified calculation - more sophisticated methods would use proper geodesy
            lat_offset = (horizontal_dist * np.cos(np.radians(sat_az))) / 111320  # meters per degree latitude
            lon_offset = (horizontal_dist * np.sin(np.radians(sat_az))) / (
                111320 * np.cos(np.radians(self.antenna_lat)))

            sample_lat = self.antenna_lat + lat_offset
            sample_lon = self.antenna_lon + lon_offset

            # Get terrain elevation at this point
            terrain_elev = self.get_terrain_elevation(sample_lat, sample_lon)

            if terrain_elev is not None:
                # Calculate required elevation angle to clear terrain
                # The terrain height relative to antenna elevation
                terrain_height_above_antenna = terrain_elev - self.antenna_elevation

                # Calculate the elevation angle needed to clear this terrain point
                required_elevation = np.degrees(
                    np.arctan2(terrain_height_above_antenna, horizontal_dist))

                # Use apparent elevation (with atmospheric refraction) for comparison
                if apparent_elevation < required_elevation:
                    return False, terrain_elev, {
                        'apparent_elevation': apparent_elevation,
                        'atmospheric_delay': atmospheric_delay,
                        'water_vapor_absorption': water_vapor_absorption,
                        'water_vapor_emission': water_vapor_emission,
                        'total_atmospheric_loss': total_atmospheric_loss,
                        'required_elevation': required_elevation,
                        'blocking_distance': dist,
                        'terrain_height_above_antenna': terrain_height_above_antenna
                    }

        return True, 0, {
            'apparent_elevation': apparent_elevation,
            'atmospheric_delay': atmospheric_delay,
            'water_vapor_absorption': water_vapor_absorption,
            'water_vapor_emission': water_vapor_emission,
            'total_atmospheric_loss': total_atmospheric_loss
        }

    def apply_terrain_masking(self, sat_alt, sat_az, sat_range, frequency=11e9):
        """
        Apply comprehensive terrain masking with integrated atmospheric effects to satellite visibility

        Args:
            sat_alt: Satellite elevation angle in degrees
            sat_az: Satellite azimuth in degrees
            sat_range: Range to satellite in meters
            frequency: Observation frequency in Hz

        Returns:
            masking_factor: Factor to apply to signal (0 = blocked, 1 = clear)
            atmospheric_effects: Dictionary with atmospheric effect details
        """
        # Check antenna limitations first
        antenna_ok = self.check_antenna_limitations(sat_alt)
        if not antenna_ok:
            return 0.0, {}

        # Check elevation masking (using advanced atmospheric refraction)
        elevation_ok = self.check_elevation_masking(sat_alt)
        if not elevation_ok:
            return 0.0, {}

        # Check line of sight with integrated atmospheric effects
        is_visible, blocking_elevation, atmospheric_effects = self.check_line_of_sight(
            sat_alt, sat_az, sat_range, frequency)

        if is_visible:
            return 1.0, atmospheric_effects
        else:
            return 0.0, atmospheric_effects  # Completely blocked

    def apply_terrain_masking_vectorized(self, sat_alts, sat_azs, sat_ranges, frequency=11e9):
        """
        VECTORIZED terrain masking for multiple satellites at once

        Args:
            sat_alts: Array of satellite elevation angles in degrees
            sat_azs: Array of satellite azimuths in degrees
            sat_ranges: Array of satellite ranges in meters
            frequency: Observation frequency in Hz

        Returns:
            masking_factors: Array of masking factors (0 = blocked, 1 = clear)
        """
        n_satellites = len(sat_alts)
        masking_factors = np.zeros(n_satellites)

        # Vectorized antenna limitations check
        antenna_ok = sat_alts >= self.antenna_mechanical_limit
        masking_factors[~antenna_ok] = 0.0

        # Vectorized elevation masking check
        elevation_ok = sat_alts >= self.min_elevation_angle
        masking_factors[~elevation_ok] = 0.0

        # For satellites that pass basic checks, apply terrain ray tracing
        valid_mask = antenna_ok & elevation_ok
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # ENABLED: Terrain ray tracing with optimizations
            # OPTIMIZATION: Skip terrain ray tracing for high elevation satellites
            high_elev_mask = sat_alts[valid_indices] > 30.0
            masking_factors[valid_indices[high_elev_mask]] = 1.0

            # Apply terrain ray tracing only to low elevation satellites
            low_elev_indices = valid_indices[~high_elev_mask]

            if len(low_elev_indices) > 0:
                # Process low elevation satellites in batches for memory efficiency
                batch_size = 1000
                for i in range(0, len(low_elev_indices), batch_size):
                    batch_end = min(i + batch_size, len(low_elev_indices))
                    batch_indices = low_elev_indices[i:batch_end]

                    # Vectorized terrain ray tracing for this batch
                    batch_factors = self._vectorized_terrain_ray_tracing(
                        sat_alts[batch_indices],
                        sat_azs[batch_indices],
                        sat_ranges[batch_indices]
                    )
                    masking_factors[batch_indices] = batch_factors

        return masking_factors

    def _vectorized_terrain_ray_tracing(self, elevations, azimuths, ranges):
        """
        ULTRA-FAST terrain ray tracing with pre-computed terrain grid
        """
        n_satellites = len(elevations)
        factors = np.ones(n_satellites)

        # AGGRESSIVE OPTIMIZATION: Skip terrain ray tracing for most satellites
        # Only check satellites that are very close to horizon (elevation < 25°)
        low_elev_mask = elevations < 25.0
        factors[~low_elev_mask] = 1.0  # Assume clear for higher elevations

        low_elev_indices = np.where(low_elev_mask)[0]
        if len(low_elev_indices) == 0:
            return factors

        # For low elevation satellites, use SMART SAMPLING based on elevation
        for i in low_elev_indices:
            elevation = elevations[i]
            azimuth = azimuths[i]

            # SMART SAMPLING: More points for very low elevation satellites
            if elevation < 15:
                # Very low elevation: 5 points for maximum accuracy
                key_distances = [500, 1000, 2000, 3000, 5000]
            elif elevation < 25:
                # Low elevation: 3 points for good accuracy
                key_distances = [1000, 2500, 4000]
            else:
                # Medium-low elevation: 2 points for speed
                key_distances = [1500, 3500]

            for dist in key_distances:
                # Calculate position along line of sight
                horizontal_dist = dist * np.cos(np.radians(elevation))

                # Convert to geographic coordinates
                lat_offset = (horizontal_dist * np.cos(np.radians(azimuth))) / 111320
                lon_offset = (horizontal_dist * np.sin(np.radians(azimuth))) / (
                    111320 * np.cos(np.radians(self.antenna_lat)))

                sample_lat = self.antenna_lat + lat_offset
                sample_lon = self.antenna_lon + lon_offset

                # ULTRA-FAST: Use simplified terrain lookup
                terrain_elev = self._fast_terrain_lookup(sample_lat, sample_lon)

                if terrain_elev is not None:
                    terrain_height_above_antenna = terrain_elev - self.antenna_elevation
                    required_elevation = np.degrees(
                        np.arctan2(terrain_height_above_antenna, horizontal_dist))

                    # Use true elevation for comparison
                    if elevation < required_elevation:
                        factors[i] = 0.0  # Blocked
                        break

        return factors

    def _fast_terrain_lookup(self, lat, lon):
        """
        ULTRA-FAST terrain elevation lookup using pre-computed grid
        """
        # Check if we have a pre-computed terrain grid
        if not hasattr(self, '_terrain_grid') or self._terrain_grid is None:
            self._build_terrain_grid()

        # Convert lat/lon to grid indices
        lat_idx = int((lat - self._grid_lat_min) / self._grid_lat_step)
        lon_idx = int((lon - self._grid_lon_min) / self._grid_lon_step)

        # Check bounds
        if (0 <= lat_idx < self._grid_lat_size and
                0 <= lon_idx < self._grid_lon_size):
            return self._terrain_grid[lat_idx, lon_idx]
        else:
            return None

    def _build_terrain_grid(self):
        """
        Pre-compute a coarse terrain grid for ultra-fast lookups
        """
        print("    Building fast terrain lookup grid...")

        # Create a coarse grid around the antenna (5km x 5km, 100m resolution)
        grid_size = 50  # 50x50 grid = 2500 points
        grid_range = 2500  # 2.5km radius

        self._grid_lat_min = self.antenna_lat - grid_range / 111320
        self._grid_lat_max = self.antenna_lat + grid_range / 111320
        self._grid_lon_min = self.antenna_lon - grid_range / (111320 * np.cos(np.radians(self.antenna_lat)))
        self._grid_lon_max = self.antenna_lon + grid_range / (111320 * np.cos(np.radians(self.antenna_lat)))

        self._grid_lat_step = (self._grid_lat_max - self._grid_lat_min) / grid_size
        self._grid_lon_step = (self._grid_lon_max - self._grid_lon_min) / grid_size
        self._grid_lat_size = grid_size
        self._grid_lon_size = grid_size

        # Pre-compute terrain elevations for the entire grid
        self._terrain_grid = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                lat = self._grid_lat_min + i * self._grid_lat_step
                lon = self._grid_lon_min + j * self._grid_lon_step
                self._terrain_grid[i, j] = self.get_terrain_elevation(lat, lon)

        print(f"    Terrain grid built: {grid_size}x{grid_size} points")
