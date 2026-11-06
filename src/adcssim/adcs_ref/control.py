"""Magnetorquer control laws (thesis §4.2.3)."""

import numpy as np
from numpy.typing import NDArray

from adcssim.frames import quat_conj, quat_mul

FloatArray = NDArray[np.float64]


def bdot(b_dot: FloatArray, gain: float) -> FloatArray:
    """B-dot detumbling, eq. 114: ``m = −k Ḃ`` (Ḃ in the body frame, T/s)."""
    return -gain * b_dot


def quaternion_feedback(
    q: FloatArray, q_target: FloatArray, w: FloatArray, kp: float, kd: float
) -> FloatArray:
    """Quaternion-feedback PD torque, eq. 119: ``τ = −kp sign(δq4) δq1:3 − kd ω``.

    ``δq = q ⊗ q_d⁻¹`` (eq. 116). The sign term makes the controller take the shorter of
    the two rotations to the target.
    """
    dq = quat_mul(q, quat_conj(q_target))
    sign = 1.0 if dq[3] >= 0.0 else -1.0
    return np.asarray(-kp * sign * dq[:3] - kd * w)


def torque_to_dipole(torque: FloatArray, b_body: FloatArray) -> FloatArray:
    """Eq. 121: ``m = (B × τ) / |B|²``. Only the torque component perpendicular to B can be
    produced by magnetorquers; this gives the dipole that produces exactly that component."""
    b2 = float(b_body @ b_body)
    if b2 == 0.0:
        return np.zeros(3)
    return np.asarray(np.cross(b_body, torque) / b2)


def saturate(m: FloatArray, max_dipole: FloatArray) -> tuple[FloatArray, bool]:
    """Scale the whole dipole vector so no coil exceeds its limit, keeping its direction."""
    ratio = float(np.max(np.abs(m) / max_dipole))
    if ratio <= 1.0:
        return m, False
    return m / ratio, True
