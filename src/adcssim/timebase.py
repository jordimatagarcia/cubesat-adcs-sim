"""Time conversions: UTC, Julian dates, decimal years and Greenwich mean sidereal time.

Julian dates are carried as a pair ``(jd, fr)``: ``jd`` is the Julian date at the preceding
midnight (it ends in .5) and ``fr`` is the fraction of the day. Splitting them keeps
sub-millisecond precision, and it's the form the SGP4 propagator expects.

UT1 is approximated by UTC. The difference is under 0.9 s, which rotates the Earth by less
than 0.004 degrees: negligible for attitude work.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

SECONDS_PER_DAY = 86_400.0
JD_J2000 = 2_451_545.0
"""Julian date of the J2000.0 epoch (2000-01-01 12:00 TT, used here as UT1)."""


def _require_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return t.astimezone(UTC)


def julian_date(t: datetime) -> tuple[float, float]:
    """Julian date of a UTC datetime, as ``(jd at midnight, day fraction)``.

    Valid for 1900-03-01 to 2100-02-28 (Vallado, *Fundamentals of Astrodynamics*, alg. 14).
    """
    t = _require_utc(t)
    y, m, d = t.year, t.month, t.day
    jd = 367 * y - (7 * (y + (m + 9) // 12)) // 4 + (275 * m) // 9 + d + 1_721_013.5
    seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6
    return float(jd), seconds / SECONDS_PER_DAY


def from_julian_date(jd: float, fr: float = 0.0) -> datetime:
    """UTC datetime of a Julian date (inverse of :func:`julian_date`)."""
    j2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)
    days = (jd - JD_J2000) + fr
    return j2000 + timedelta(days=days)


def decimal_year(t: datetime) -> float:
    """Year with fraction, e.g. 2021-07-02 12:00 UTC → 2021.5. Used by the IGRF model."""
    t = _require_utc(t)
    start = datetime(t.year, 1, 1, tzinfo=UTC)
    end = datetime(t.year + 1, 1, 1, tzinfo=UTC)
    return t.year + (t - start) / (end - start)


def gmst(jd: ArrayLike, fr: ArrayLike = 0.0) -> NDArray[np.float64]:
    """Greenwich mean sidereal time in radians, in [0, 2π), IAU-82 model.

    This is the angle that rotates the TEME frame (SGP4 output) into the Earth-fixed frame
    (Vallado, eq. 3-47).
    """
    jd_arr = np.asarray(jd, dtype=np.float64)
    fr_arr = np.asarray(fr, dtype=np.float64)
    t = ((jd_arr - JD_J2000) + fr_arr) / 36_525.0
    seconds = (
        67_310.54841 + (876_600.0 * 3600.0 + 8_640_184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    )
    angle = np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY * 2.0 * math.pi
    return np.asarray(angle, dtype=np.float64)


@dataclass(frozen=True)
class Clock:
    """Maps simulation time (seconds since ``start``) to absolute time."""

    start: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _require_utc(self.start))

    @property
    def jd0(self) -> tuple[float, float]:
        return julian_date(self.start)

    def jd(self, t: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Julian date pair at simulation time ``t`` (seconds). ``t`` may be an array."""
        jd, fr = self.jd0
        return jd, fr + np.asarray(t, dtype=np.float64) / SECONDS_PER_DAY

    def datetime(self, t: float) -> datetime:
        return self.start + timedelta(seconds=float(t))

    def decimal_year(self, t: float) -> float:
        return decimal_year(self.datetime(t))
