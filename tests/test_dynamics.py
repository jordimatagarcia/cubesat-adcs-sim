import math

import numpy as np
import pytest

from adcssim.dynamics import inertial_angular_momentum, kinetic_energy, rk4_step
from adcssim.frames import quat_from_axis_angle, quat_mul, quat_normalize, quat_positive
from adcssim.spacecraft import Spacecraft

J = np.diag([0.0418, 0.0418, 0.0067])
J_ASYM = np.array([[0.050, 0.001, -0.002], [0.001, 0.040, 0.0005], [-0.002, 0.0005, 0.010]])


def no_torque(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.zeros(3)


def propagate(
    q: np.ndarray, w: np.ndarray, inertia: np.ndarray, dt: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(steps):
        q, w = rk4_step(q, w, dt, inertia, no_torque)
    return q, w


def test_torque_free_motion_conserves_energy_and_momentum() -> None:
    q0 = quat_normalize(np.array([0.1, -0.3, 0.2, 0.9]))
    w0 = np.radians([3.0, 5.0, -4.0])
    q, w = propagate(q0, w0, J_ASYM, 0.1, 36_000)  # one hour
    assert kinetic_energy(w, J_ASYM) == pytest.approx(kinetic_energy(w0, J_ASYM), rel=1e-9)
    np.testing.assert_allclose(
        inertial_angular_momentum(q, w, J_ASYM),
        inertial_angular_momentum(q0, w0, J_ASYM),
        rtol=1e-8,
    )


def test_axisymmetric_body_matches_analytic_solution() -> None:
    # J = diag(I, I, I3): ω3 is constant and (ω1, ω2) rotates at Ω = (I3 − I)/I · ω3.
    i_t, i_3 = 0.0418, 0.0067
    w0 = np.array([0.02, -0.01, 0.15])
    omega = (i_3 - i_t) / i_t * w0[2]
    t_end = 600.0
    _, w = propagate(np.array([0.0, 0.0, 0.0, 1.0]), w0, np.diag([i_t, i_t, i_3]), 0.1, 6000)
    c, s = math.cos(omega * t_end), math.sin(omega * t_end)
    expected = [w0[0] * c - w0[1] * s, w0[0] * s + w0[1] * c, w0[2]]
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_constant_rate_kinematics_matches_closed_form() -> None:
    # Spherical inertia: ω stays constant and q(t) = δq(ω t) ⊗ q0.
    q0 = quat_normalize(np.array([0.3, 0.1, -0.2, 0.9]))
    w = np.radians([10.0, 10.0, 10.0])
    t_end = 100.0
    q, _ = propagate(q0, w, np.eye(3) * 0.03, 0.1, 1000)
    rate = float(np.linalg.norm(w))
    expected = quat_mul(quat_from_axis_angle(w / rate, rate * t_end), q0)
    np.testing.assert_allclose(quat_positive(q), quat_positive(expected), atol=1e-10)


def test_rk4_is_fourth_order() -> None:
    q0 = quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    w0 = np.array([0.5, -0.3, 0.8])
    _, w_ref = propagate(q0, w0, J_ASYM, 0.0125, 1600)  # 20 s
    errors = []
    for dt, steps in ((0.2, 100), (0.1, 200)):
        _, w = propagate(q0, w0, J_ASYM, dt, steps)
        errors.append(np.linalg.norm(w - w_ref))
    # Halving the step divides the error by 2⁴.
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.25)


def test_constant_torque_spins_up_linearly() -> None:
    sc = Spacecraft()
    tau = np.array([0.0, 0.0, 1e-6])

    def torque(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        return tau

    q, w = np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
    for _ in range(100):
        q, w = rk4_step(q, w, 0.1, sc.inertia, torque)
    # About the z principal axis there is no gyroscopic coupling: ω_z = τ t / I_z.
    assert w[2] == pytest.approx(1e-6 * 10.0 / sc.inertia[2, 2], rel=1e-12)
    assert np.allclose(w[:2], 0.0)
