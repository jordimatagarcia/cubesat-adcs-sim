"""Plausibility limits for sensor readings and on-board reference vectors.

A reading outside these limits is treated like a NaN: ignored and flagged ``BAD_INPUT``; an
out-of-range reference vector is treated as unavailable. The limits are far beyond anything
a working sensor reports in orbit. They only stop absurd values (a corrupted packet, a
failed sensor) from overflowing the filter state, which fuzzing the C library showed can
otherwise produce a finite but meaningless all-zero quaternion. The C and Rust libraries
use the same values.
"""

import numpy as np
from numpy.typing import NDArray

GYRO_MAX = 50.0
"""rad/s. MEMS gyros saturate at about 35 rad/s (2000 °/s)."""
FIELD_MAX = 1e-3
"""T. The geomagnetic field in low Earth orbit stays below 7e-5 T."""
UNIT_MIN, UNIT_MAX = 0.5, 2.0
"""Accepted norm of a direction that should be a unit vector (sun sensor, ephemeris)."""


def _norm_if_finite(v: NDArray[np.float64]) -> float | None:
    a = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        return None
    return float(np.linalg.norm(a))


def gyro_plausible(v: NDArray[np.float64]) -> bool:
    n = _norm_if_finite(v)
    return n is not None and n <= GYRO_MAX


def field_plausible(v: NDArray[np.float64]) -> bool:
    n = _norm_if_finite(v)
    return n is not None and 0.0 < n <= FIELD_MAX


def unit_plausible(v: NDArray[np.float64]) -> bool:
    n = _norm_if_finite(v)
    return n is not None and UNIT_MIN <= n <= UNIT_MAX


def unit(v: object) -> NDArray[np.float64]:
    """``v`` scaled to unit length. Directions are normalised after the plausibility check,
    which accepts lengths 0.5–2 (FLAE and the vector measurements assume unit vectors)."""
    a = np.asarray(v, dtype=np.float64)
    return np.asarray(a / np.linalg.norm(a), dtype=np.float64)


def safe_norm(n: float) -> float:
    """``n`` if it can be divided by, NaN otherwise (zero or overflowed), so a degenerate
    normalisation turns into NaN and the filter's finiteness check catches it."""
    return n if 0.0 < n < np.inf else float("nan")
