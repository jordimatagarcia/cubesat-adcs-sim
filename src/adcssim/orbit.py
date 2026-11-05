"""Orbit propagation with SGP4 (Vallado et al. 2006, via the ``sgp4`` package).

Positions and velocities are in the TEME frame, which this project uses as its inertial
frame, in metres and metres per second.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

from adcssim.timebase import Clock, from_julian_date, julian_date

FloatArray = NDArray[np.float64]

MU_EARTH_WGS72 = 3.986008e14
"""Earth gravitational parameter of WGS-72 (m³/s²), the constants SGP4 is defined with."""
R_EARTH_WGS72 = 6_378_135.0
"""WGS-72 equatorial radius (m)."""

_EPOCH_1949 = 2_433_281.5
"""Julian date of 1949-12-31 00:00 UT, the reference of ``Satrec.sgp4init`` epochs."""


class OrbitError(RuntimeError):
    """SGP4 reported an error (for example, the orbit decayed)."""


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum of a TLE line (digits count their value, '-' counts 1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


@dataclass(frozen=True)
class Tle:
    """A two-line element set. The checksum of both lines is verified."""

    line1: str
    line2: str
    name: str = ""

    def __post_init__(self) -> None:
        for number, line in (("1", self.line1), ("2", self.line2)):
            if len(line) != 69 or not line.startswith(number + " "):
                raise ValueError(f"TLE line {number} is malformed: {line!r}")
            if int(line[68]) != tle_checksum(line):
                raise ValueError(f"TLE line {number} has a wrong checksum: {line!r}")

    @classmethod
    def from_file(cls, path: Path) -> "Tle":
        """Read a 2-line or 3-line (name + 2 lines) TLE file. ``#`` lines are comments."""
        lines = [
            ln.rstrip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if len(lines) == 2:
            return cls(lines[0], lines[1])
        if len(lines) == 3:
            return cls(lines[1], lines[2], name=lines[0].strip())
        raise ValueError(f"{path}: expected 2 or 3 TLE lines, found {len(lines)}")


class Orbit:
    """SGP4 propagator for one satellite."""

    def __init__(self, satrec: Satrec) -> None:
        self._sat = satrec

    @classmethod
    def from_tle(cls, tle: Tle) -> "Orbit":
        return cls(Satrec.twoline2rv(tle.line1, tle.line2, WGS72))

    @classmethod
    def from_elements(
        cls,
        epoch: datetime,
        altitude_m: float,
        inclination_deg: float,
        raan_deg: float,
        eccentricity: float = 0.0,
        arg_perigee_deg: float = 0.0,
        mean_anomaly_deg: float = 0.0,
        bstar: float = 0.0,
    ) -> "Orbit":
        """Orbit from mean elements, without a TLE.

        ``altitude_m`` sets the semi-major axis above the WGS-72 equatorial radius. The mean
        motion is computed from Kepler's third law, which is a close approximation of the
        Kozai mean motion SGP4 expects.
        """
        a = R_EARTH_WGS72 + altitude_m
        n_rad_min = math.sqrt(MU_EARTH_WGS72 / a**3) * 60.0
        jd, fr = julian_date(epoch)
        sat = Satrec()
        sat.sgp4init(
            WGS72,
            "i",
            99999,
            (jd - _EPOCH_1949) + fr,
            bstar,
            0.0,
            0.0,
            eccentricity,
            math.radians(arg_perigee_deg),
            math.radians(inclination_deg),
            math.radians(mean_anomaly_deg),
            n_rad_min,
            math.radians(raan_deg),
        )
        return cls(sat)

    @property
    def epoch(self) -> datetime:
        """Epoch of the element set (UTC)."""
        return from_julian_date(self._sat.jdsatepoch, self._sat.jdsatepochF).astimezone(UTC)

    @property
    def period_s(self) -> float:
        """Orbital period from the mean motion, in seconds."""
        return float(2.0 * math.pi / self._sat.no_kozai * 60.0)

    def state(self, jd: float, fr: float) -> tuple[FloatArray, FloatArray]:
        """Position (m) and velocity (m/s) in TEME at Julian date ``jd + fr``."""
        err, r, v = self._sat.sgp4(jd, fr)
        if err != 0:
            raise OrbitError(SGP4_ERRORS.get(err, f"SGP4 error {err}"))
        return np.array(r) * 1e3, np.array(v) * 1e3

    def states(self, jd: float, fr: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Vectorised :meth:`state` for an array of day fractions. Returns ``(N, 3)`` arrays."""
        fr_arr = np.atleast_1d(np.asarray(fr, dtype=np.float64))
        jd_arr = np.full_like(fr_arr, jd)
        err, r, v = self._sat.sgp4_array(jd_arr, fr_arr)
        if np.any(err != 0):
            code = int(err[np.nonzero(err)[0][0]])
            raise OrbitError(SGP4_ERRORS.get(code, f"SGP4 error {code}"))
        return np.asarray(r) * 1e3, np.asarray(v) * 1e3

    def predict(
        self, clock: Clock, t0: float, duration_s: float, step_s: float = 30.0
    ) -> tuple[FloatArray, FloatArray]:
        """Positions over ``[t0, t0 + duration_s]`` (simulation seconds), e.g. the next orbit.

        Returns the sample times and the ``(N, 3)`` TEME positions.
        """
        t = np.arange(t0, t0 + duration_s + 0.5 * step_s, step_s)
        jd, fr = clock.jd(t)
        r, _ = self.states(jd, fr)
        return t, r
