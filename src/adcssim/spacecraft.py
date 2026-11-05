"""Physical parameters of the 3U CubeSat."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

U3_DIMENSIONS_M = (0.1, 0.1, 0.34)
"""Outer dimensions of a 3U CubeSat along body x, y, z (m)."""


def box_inertia(mass_kg: float, dims_m: tuple[float, float, float]) -> FloatArray:
    """Inertia tensor (kg·m²) of a uniform box about its centre of mass."""
    x, y, z = dims_m
    return np.diag(
        [
            mass_kg / 12.0 * (y * y + z * z),
            mass_kg / 12.0 * (x * x + z * z),
            mass_kg / 12.0 * (x * x + y * y),
        ]
    )


def _default_inertia() -> FloatArray:
    return box_inertia(4.0, U3_DIMENSIONS_M)


@dataclass(frozen=True)
class Spacecraft:
    """Mass properties and surface model.

    The defaults describe a generic 4 kg 3U CubeSat with uniform mass distribution. They are
    not the Foresail-1 values, which are not published.
    """

    mass_kg: float = 4.0
    inertia: FloatArray = field(default_factory=_default_inertia)
    """Inertia tensor about the centre of mass, body frame (kg·m²)."""
    dimensions_m: tuple[float, float, float] = U3_DIMENSIONS_M
    residual_dipole: FloatArray = field(default_factory=lambda: np.full(3, 1e-3))
    """Residual magnetic dipole, body frame (A·m²)."""
    centre_of_pressure: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.005]))
    """Centre of pressure relative to the centre of mass, body frame (m)."""
    drag_coefficient: float = 2.2
    reflectivity_coefficient: float = 1.3
    """Solar radiation pressure coefficient C_r (1 = absorber, 2 = mirror)."""

    def __post_init__(self) -> None:
        j = np.asarray(self.inertia, dtype=np.float64)
        if j.shape != (3, 3) or not np.allclose(j, j.T):
            raise ValueError("inertia must be a symmetric 3x3 matrix")
        if np.any(np.linalg.eigvalsh(j) <= 0.0):
            raise ValueError("inertia must be positive definite")
        object.__setattr__(self, "inertia", j)
        object.__setattr__(
            self, "residual_dipole", np.asarray(self.residual_dipole, dtype=np.float64)
        )
        object.__setattr__(
            self, "centre_of_pressure", np.asarray(self.centre_of_pressure, dtype=np.float64)
        )

    @property
    def face_areas(self) -> FloatArray:
        """Area of the faces normal to body x, y, z (m²)."""
        x, y, z = self.dimensions_m
        return np.array([y * z, x * z, x * y])

    def projected_area(self, direction_b: FloatArray) -> float:
        """Cross-section (m²) seen along a body-frame unit vector."""
        return float(np.sum(self.face_areas * np.abs(direction_b)))
