"""Environmental disturbance torques in low Earth orbit (thesis §3.4).

Every function returns a body-frame torque in N·m. Inputs are body-frame vectors unless
noted.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from adcssim.environment import AU, SOLAR_PRESSURE_1AU
from adcssim.orbit import MU_EARTH_WGS72
from adcssim.spacecraft import Spacecraft

FloatArray = NDArray[np.float64]


def gravity_gradient(r_b: FloatArray, inertia: FloatArray) -> FloatArray:
    """``τ = 3μ/|r|⁵ · r × (J r)`` with ``r`` the position vector in the body frame."""
    rn = float(np.linalg.norm(r_b))
    return np.asarray(3.0 * MU_EARTH_WGS72 / rn**5 * np.cross(r_b, inertia @ r_b))


def residual_dipole(m_b: FloatArray, b_b: FloatArray) -> FloatArray:
    """Torque of a magnetic dipole ``m`` (A·m²) in the field ``B`` (T): ``τ = m × B``."""
    return np.asarray(np.cross(m_b, b_b))


def aerodynamic(sc: Spacecraft, v_rel_b: FloatArray, density: float) -> FloatArray:
    """Drag force ``-½ ρ C_d A |v| v`` applied at the centre of pressure."""
    speed = float(np.linalg.norm(v_rel_b))
    if speed == 0.0:
        return np.zeros(3)
    area = sc.projected_area(v_rel_b / speed)
    force = -0.5 * density * sc.drag_coefficient * area * speed * v_rel_b
    return np.asarray(np.cross(sc.centre_of_pressure, force))


def solar_pressure(
    sc: Spacecraft, sun_b: FloatArray, sun_distance: float, eclipse: bool
) -> FloatArray:
    """Solar radiation pressure force along ``-ŝ`` applied at the centre of pressure."""
    if eclipse:
        return np.zeros(3)
    pressure = SOLAR_PRESSURE_1AU * (AU / sun_distance) ** 2
    force = -pressure * sc.reflectivity_coefficient * sc.projected_area(sun_b) * sun_b
    return np.asarray(np.cross(sc.centre_of_pressure, force))


@dataclass(frozen=True)
class DisturbanceModel:
    """Which disturbance torques are included in the simulation."""

    gravity_gradient: bool = True
    residual_dipole: bool = True
    aerodynamic: bool = True
    solar_pressure: bool = True

    def torque(
        self,
        sc: Spacecraft,
        r_b: FloatArray,
        b_b: FloatArray,
        v_rel_b: FloatArray,
        density: float,
        sun_b: FloatArray,
        sun_distance: float,
        eclipse: bool,
    ) -> FloatArray:
        total = np.zeros(3)
        if self.gravity_gradient:
            total += gravity_gradient(r_b, sc.inertia)
        if self.residual_dipole:
            total += residual_dipole(sc.residual_dipole, b_b)
        if self.aerodynamic:
            total += aerodynamic(sc, v_rel_b, density)
        if self.solar_pressure:
            total += solar_pressure(sc, sun_b, sun_distance, eclipse)
        return total
