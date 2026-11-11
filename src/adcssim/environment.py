"""Space environment along the orbit: position, Earth rotation, magnetic field, Sun, eclipse
and atmospheric density, evaluated together for a batch of simulation times."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adcssim.frames import (
    ecef_to_geodetic,
    ecef_to_teme,
    teme_to_ecef,
    teme_to_ecef_velocity,
    unit,
)
from adcssim.igrf import Igrf
from adcssim.orbit import Orbit
from adcssim.timebase import JD_J2000, Clock, gmst

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

AU = 149_597_870_700.0
"""Astronomical unit (m)."""
R_EARTH = 6_378_137.0
"""Earth equatorial radius used for the shadow model (m)."""
SOLAR_PRESSURE_1AU = 4.56e-6
"""Solar radiation pressure on a perfect absorber at 1 AU (N/m²)."""


def sun_position(jd: ArrayLike, fr: ArrayLike = 0.0) -> FloatArray:
    """Earth→Sun vector (m) in the mean-of-date equatorial frame.

    Low-precision almanac model (Vallado, alg. 29), accurate to about 0.01°. The difference
    between mean-of-date and TEME (nutation, under 0.01°) is neglected, so the result is used
    directly in TEME.
    """
    t = ((np.asarray(jd, dtype=np.float64) - JD_J2000) + np.asarray(fr, dtype=np.float64)) / (
        36_525.0
    )
    mean_lon = np.radians(280.460 + 36_000.771 * t)
    mean_anom = np.radians(357.529_109_2 + 35_999.050_34 * t)
    ecl_lon = mean_lon + np.radians(
        1.914_666_471 * np.sin(mean_anom) + 0.019_994_643 * np.sin(2.0 * mean_anom)
    )
    dist = (
        1.000_140_612 - 0.016_708_617 * np.cos(mean_anom) - 0.000_139_589 * np.cos(2.0 * mean_anom)
    )
    obliquity = np.radians(23.439_291 - 0.013_004_2 * t)
    r = dist * AU
    return np.stack(
        [
            r * np.cos(ecl_lon),
            r * np.cos(obliquity) * np.sin(ecl_lon),
            r * np.sin(obliquity) * np.sin(ecl_lon),
        ],
        axis=-1,
    )


def in_eclipse(r_sat: ArrayLike, r_sun: ArrayLike) -> BoolArray:
    """Cylindrical Earth-shadow test (thesis eqs. 122–125).

    The satellite is in eclipse when it is on the night side (``r · ŝ < 0``) and its distance
    from the Earth–Sun line is less than the Earth radius.
    """
    r = np.asarray(r_sat, dtype=np.float64)
    s = unit(r_sun)
    along = np.sum(r * s, axis=-1)
    perp = np.linalg.norm(r - along[..., None] * s, axis=-1)
    return np.asarray((along < 0.0) & (perp < R_EARTH))


# Exponential atmosphere, Vallado (2013) table 8-4: base altitude (km), density (kg/m³),
# scale height (km).
_ATMOSPHERE = np.array(
    [
        (0, 1.225, 7.249),
        (25, 3.899e-2, 6.349),
        (30, 1.774e-2, 6.682),
        (40, 3.972e-3, 7.554),
        (50, 1.057e-3, 8.382),
        (60, 3.206e-4, 7.714),
        (70, 8.770e-5, 6.549),
        (80, 1.905e-5, 5.799),
        (90, 3.396e-6, 5.382),
        (100, 5.297e-7, 5.877),
        (110, 9.661e-8, 7.263),
        (120, 2.438e-8, 9.473),
        (130, 8.484e-9, 12.636),
        (140, 3.845e-9, 16.149),
        (150, 2.070e-9, 22.523),
        (180, 5.464e-10, 29.740),
        (200, 2.789e-10, 37.105),
        (250, 7.248e-11, 45.546),
        (300, 2.418e-11, 53.628),
        (350, 9.518e-12, 53.298),
        (400, 3.725e-12, 58.515),
        (450, 1.585e-12, 60.828),
        (500, 6.967e-13, 63.822),
        (600, 1.454e-13, 71.835),
        (700, 3.614e-14, 88.667),
        (800, 1.170e-14, 124.64),
        (900, 5.245e-15, 181.05),
        (1000, 3.019e-15, 268.00),
    ]
)


def atmospheric_density(altitude_m: ArrayLike) -> FloatArray:
    """Exponential-model atmospheric density (kg/m³) at geodetic altitude(s) in metres."""
    h_km = np.asarray(altitude_m, dtype=np.float64) / 1e3
    idx = np.clip(np.searchsorted(_ATMOSPHERE[:, 0], h_km, side="right") - 1, 0, None)
    h0, rho0, scale = _ATMOSPHERE[idx, 0], _ATMOSPHERE[idx, 1], _ATMOSPHERE[idx, 2]
    return np.asarray(rho0 * np.exp(-(h_km - h0) / scale), dtype=np.float64)


@dataclass(frozen=True)
class EnvironmentSample:
    """Environment at ``N`` simulation times. Vectors are ``(N, 3)``, in TEME unless noted."""

    t: FloatArray
    r: FloatArray
    """Satellite position (m)."""
    v: FloatArray
    """Satellite velocity (m/s)."""
    v_rel: FloatArray
    """Velocity relative to the co-rotating atmosphere (m/s)."""
    gmst: FloatArray
    """Greenwich mean sidereal time (rad)."""
    b: FloatArray
    """Geomagnetic field (T)."""
    sun: FloatArray
    """Satellite→Sun unit vector."""
    sun_distance: FloatArray
    """Satellite→Sun distance (m)."""
    eclipse: BoolArray
    lat: FloatArray
    """Geodetic latitude (rad)."""
    lon: FloatArray
    """Longitude (rad)."""
    alt: FloatArray
    """Geodetic altitude (m)."""
    density: FloatArray
    """Atmospheric density (kg/m³)."""

    def at(self, i: int) -> "EnvironmentSample":
        """The ``i``-th sample as a batch of one."""
        return self.slice(i, i + 1)

    def slice(self, start: int, stop: int) -> "EnvironmentSample":
        """Samples ``start`` to ``stop - 1``."""
        sl = slice(start, stop)
        return EnvironmentSample(
            **{name: getattr(self, name)[sl] for name in self.__dataclass_fields__}
        )


class Environment:
    """Evaluates the environment models for one orbit."""

    def __init__(self, orbit: Orbit, clock: Clock, igrf: Igrf | None = None) -> None:
        self.orbit = orbit
        self.clock = clock
        self.igrf = igrf if igrf is not None else Igrf()
        self._year0 = clock.decimal_year(0.0)

    def evaluate(self, t: ArrayLike) -> EnvironmentSample:
        """Environment at simulation time(s) ``t`` (seconds since the clock start)."""
        tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
        jd, fr = self.clock.jd(tt)
        r, v = self.orbit.states(jd, fr)
        theta = gmst(jd, fr)
        r_ecef = teme_to_ecef(r, theta)
        # Secular variation is at most ~0.3 nT per day, so one decimal year per batch is enough.
        year = self._year0 + float(tt.mean()) / (365.25 * 86_400.0)
        b = ecef_to_teme(self.igrf.field_ecef(r_ecef, year), theta)

        sun_from_earth = sun_position(jd, fr)
        sat_to_sun = sun_from_earth - r
        sun_dist = np.linalg.norm(sat_to_sun, axis=-1)
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        v_rel = ecef_to_teme(teme_to_ecef_velocity(r, v, theta), theta)
        return EnvironmentSample(
            t=tt,
            r=r,
            v=v,
            v_rel=v_rel,
            gmst=theta,
            b=b,
            sun=sat_to_sun / sun_dist[:, None],
            sun_distance=sun_dist,
            eclipse=in_eclipse(r, sun_from_earth),
            lat=lat,
            lon=lon,
            alt=alt,
            density=atmospheric_density(alt),
        )
