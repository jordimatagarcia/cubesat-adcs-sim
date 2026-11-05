"""Rigid-body attitude dynamics and kinematics, integrated with fourth-order Runge–Kutta.

State: attitude quaternion ``q`` (inertial → body, scalar-last) and body angular rate ``ω``
(rad/s, body relative to inertial, in body coordinates).

- Kinematics: ``q̇ = ½ Ξ(q) ω`` (Markley & Crassidis eq. 3.20)
- Dynamics: ``J ω̇ = τ − ω × (J ω)`` (Euler's equations, thesis §3.3)
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from adcssim.frames import cross3, quat_normalize, quat_to_dcm, quat_xi

FloatArray = NDArray[np.float64]
TorqueFn = Callable[[FloatArray, FloatArray], FloatArray]
"""Body torque (N·m) as a function of ``(q, ω)``."""


def derivatives(
    q: FloatArray, w: FloatArray, inertia: FloatArray, inertia_inv: FloatArray, torque: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Time derivatives ``(q̇, ω̇)``."""
    dq = 0.5 * quat_xi(q) @ w
    dw = inertia_inv @ (torque - cross3(w, inertia @ w))
    return dq, dw


def rk4_step(
    q: FloatArray,
    w: FloatArray,
    dt: float,
    inertia: FloatArray,
    torque_fn: TorqueFn,
    inertia_inv: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Advance ``(q, ω)`` by ``dt`` seconds. The quaternion is renormalised afterwards."""
    j_inv = np.linalg.inv(inertia) if inertia_inv is None else inertia_inv

    def f(qq: FloatArray, ww: FloatArray) -> tuple[FloatArray, FloatArray]:
        return derivatives(qq, ww, inertia, j_inv, torque_fn(qq, ww))

    k1q, k1w = f(q, w)
    k2q, k2w = f(q + 0.5 * dt * k1q, w + 0.5 * dt * k1w)
    k3q, k3w = f(q + 0.5 * dt * k2q, w + 0.5 * dt * k2w)
    k4q, k4w = f(q + dt * k3q, w + dt * k3w)
    q_next = q + dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    w_next = w + dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    return quat_normalize(q_next), w_next


def kinetic_energy(w: FloatArray, inertia: FloatArray) -> float:
    return float(0.5 * w @ inertia @ w)


def inertial_angular_momentum(q: FloatArray, w: FloatArray, inertia: FloatArray) -> FloatArray:
    """Angular momentum in the inertial frame, ``Aᵀ(q) J ω``: constant without torques."""
    return np.asarray(quat_to_dcm(q).T @ (inertia @ w))
