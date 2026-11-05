"""Magnetorquer model (thesis §4.1.3, Table 3)."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

FS1_MAX_DIPOLE = (0.13, 0.13, 0.082)
"""Maximum dipole of the Foresail-1 coils along x, y, z (A·m², thesis Table 3)."""


@dataclass(frozen=True)
class Magnetorquers:
    """Three orthogonal coils along the body axes.

    A command is clipped to each coil's maximum dipole and quantised to ``levels`` steps per
    polarity (PWM resolution). With a duty cycle below 1, the coils are on for that fraction
    at the start of each control period and off for the rest, so the magnetometer can sample
    without the coils' own field.
    """

    max_dipole: FloatArray = field(default_factory=lambda: np.array(FS1_MAX_DIPOLE))
    levels: int = 0
    """Quantisation steps per polarity; 0 means continuous."""
    duty_cycle: float = 1.0

    def __post_init__(self) -> None:
        m = np.asarray(self.max_dipole, dtype=np.float64)
        if m.shape != (3,) or np.any(m <= 0.0):
            raise ValueError("max_dipole must be three positive values")
        if not 0.0 < self.duty_cycle <= 1.0:
            raise ValueError("duty_cycle must be in (0, 1]")
        if self.levels < 0:
            raise ValueError("levels must be >= 0")
        object.__setattr__(self, "max_dipole", m)

    def apply(self, m_cmd: FloatArray) -> FloatArray:
        """Dipole the coils actually produce for a commanded dipole."""
        m = np.clip(np.asarray(m_cmd, dtype=np.float64), -self.max_dipole, self.max_dipole)
        if self.levels > 0:
            step = self.max_dipole / self.levels
            m = np.round(m / step) * step
        return np.asarray(m)

    def active(self, time_in_period: float, period: float) -> bool:
        """Whether the coils are on at ``time_in_period`` seconds into a control period."""
        return time_in_period < self.duty_cycle * period - 1e-12


def magnetic_torque(m_b: FloatArray, b_b: FloatArray) -> FloatArray:
    """``τ = m × B`` (thesis eq. 94)."""
    return np.asarray(np.cross(m_b, b_b))
