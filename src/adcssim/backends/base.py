"""Interface between the simulator and an ADCS implementation.

``AdcsInput`` and ``AdcsOutput`` mirror the C structs in ``adcs-c/include/adcs.h`` field for
field, so the Python reference, the C library and the Rust library are interchangeable.
The ADCS only receives what a real satellite would have: sensor readings, reference vectors
from on-board models, and time. Truth never crosses this interface.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class AdcsMode(IntEnum):
    OFF = 0
    DETUMBLE = 1
    ESTIMATE = 2
    POINT = 3
    SAFE = 4


@dataclass(frozen=True)
class AdcsContext:
    """Static information given to the ADCS when it starts."""

    dt: float
    """Control period (s)."""
    inertia: FloatArray
    """Spacecraft inertia tensor as known on board (kg·m²)."""
    max_dipole: FloatArray
    """Magnetorquer limits (A·m²)."""


@dataclass(frozen=True)
class AdcsInput:
    t: float
    """Seconds since the start of the run."""
    mag_b: FloatArray
    """Magnetometer reading, body frame (T)."""
    mag_valid: bool
    gyro_b: FloatArray
    """Gyroscope reading, body frame (rad/s)."""
    gyro_valid: bool
    sun_b: FloatArray
    """Sun direction from the sun sensors, body frame (unit vector)."""
    sun_valid: bool
    sun_ref_i: FloatArray
    """Sun direction from the on-board ephemeris, inertial frame (unit vector)."""
    mag_ref_i: FloatArray
    """Geomagnetic field from the on-board model, inertial frame (T)."""


def _nan(n: int) -> FloatArray:
    return np.full(n, np.nan)


@dataclass(frozen=True)
class AdcsOutput:
    q_est: FloatArray = field(default_factory=lambda: _nan(4))
    """Estimated attitude, scalar-last, inertial → body."""
    w_est: FloatArray = field(default_factory=lambda: _nan(3))
    """Estimated body rate (rad/s)."""
    bias_mag: FloatArray = field(default_factory=lambda: _nan(3))
    """Estimated magnetometer bias (T)."""
    bias_gyro: FloatArray = field(default_factory=lambda: _nan(3))
    """Estimated gyroscope bias (rad/s)."""
    m_cmd: FloatArray = field(default_factory=lambda: np.zeros(3))
    """Magnetorquer dipole command (A·m²)."""
    mode: AdcsMode = AdcsMode.OFF
    status: int = 0
    """Implementation-defined status flags (0 = nominal)."""


class AdcsBackend(Protocol):
    name: str

    def reset(self, ctx: AdcsContext) -> None:
        """Start (or restart) the ADCS."""

    def step(self, inp: AdcsInput) -> AdcsOutput:
        """Process one control period."""


class NoAdcs:
    """Open loop: no estimation, coils off."""

    name = "none"

    def reset(self, ctx: AdcsContext) -> None:
        pass

    def step(self, inp: AdcsInput) -> AdcsOutput:
        return AdcsOutput()


def create_backend(name: str) -> AdcsBackend:
    """Backend by name, as used in scenario files and on the command line."""
    if name == "none":
        return NoAdcs()
    raise ValueError(f"unknown ADCS backend {name!r} (available: none)")
