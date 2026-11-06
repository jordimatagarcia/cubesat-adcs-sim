"""Improved ADCS variant: USQUE filter and the assembled backend."""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.adcs_ref.adcs_improved import ImprovedAdcs
from adcssim.adcs_ref.tuning import MAG_FLOOR_UT, matched_improved_config
from adcssim.adcs_ref.usque import Usque, grp_to_quat, quat_to_grp, rotation_quat
from adcssim.backends import AdcsConfig, AdcsContext, AdcsInput, AdcsMode, AdcsStatus
from adcssim.frames import quat_angle, quat_from_axis_angle, quat_mul, quat_to_dcm
from adcssim.scenario import load_scenario
from adcssim.spacecraft import Spacecraft
from tests.conftest import REPO

SUN_I = np.array([0.6, 0.8, 0.0])
MAG_I = np.array([1e-5, -2e-5, 3e-5])


@given(arrays(np.float64, 3, elements=st.floats(-1.5, 1.5, allow_nan=False)))
def test_grp_round_trip(p: np.ndarray) -> None:
    np.testing.assert_allclose(quat_to_grp(grp_to_quat(p)), p, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(grp_to_quat(p)), 1.0)


def test_grp_equals_rotation_angle_for_small_errors() -> None:
    q = quat_from_axis_angle([0.0, 1.0, 0.0], 1e-3)
    np.testing.assert_allclose(quat_to_grp(q), [0.0, 1e-3, 0.0], rtol=1e-6)


def test_rotation_quat_matches_constant_rate_kinematics() -> None:
    w = np.radians([2.0, -3.0, 1.0])
    q0 = quat_from_axis_angle([1, 2, 3], 0.3)
    expected = quat_mul(quat_from_axis_angle(w, float(np.linalg.norm(w)) * 0.7), q0)
    np.testing.assert_allclose(quat_mul(rotation_quat(w, 0.7), q0), expected, atol=1e-14)
    np.testing.assert_allclose(rotation_quat(np.zeros(3), 0.1), [0, 0, 0, 1])


def _simulate(
    filt: Usque,
    q: np.ndarray,
    w: np.ndarray,
    gyro_bias: np.ndarray,
    mag_bias_ut: np.ndarray,
    steps: int,
    sun: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Constant-rate truth, noisy sensors; returns the final true attitude."""
    rng = np.random.default_rng(seed)
    gyro_prev = w + gyro_bias
    for _ in range(steps):
        q = quat_mul(rotation_quat(w, 0.1), q)
        a = quat_to_dcm(q)
        sun_b = a @ SUN_I + rng.normal(0, 1e-3, 3)
        mag_ut = a @ MAG_I * 1e6 + mag_bias_ut + rng.normal(0, 0.1, 3)
        filt.step(gyro_prev, 0.1, sun_b, SUN_I, sun, mag_ut, MAG_I * 1e6, True)
        gyro_prev = w + gyro_bias + rng.normal(0, 1e-4, 3)
    return q


def _filter() -> Usque:
    f = Usque(1.0, 1e-4, 1e-6, 0.1, 1e-3, 1e-3)
    return f


def test_usque_converges_from_an_initial_error() -> None:
    f = _filter()
    q_true = quat_from_axis_angle([1, -1, 2], 0.8)
    f.reset(
        quat_mul(quat_from_axis_angle([0, 0, 1], math.radians(5)), q_true),
        np.r_[np.full(3, 0.1**2), np.full(3, 1e-6), np.full(3, 1.0)],
    )
    q_end = _simulate(f, q_true, np.radians([1.0, 2.0, -1.0]), np.zeros(3), np.zeros(3), 300)
    assert math.degrees(float(quat_angle(f.q_ref, q_end))) < 0.2


def test_usque_estimates_both_biases() -> None:
    f = _filter()
    q_true = quat_from_axis_angle([0, 1, 0], 0.4)
    f.reset(q_true, np.r_[np.full(3, 0.01**2), np.full(3, 0.01**2), np.full(3, 50.0**2)])
    gyro_bias = np.radians([0.2, -0.1, 0.3])
    mag_bias = np.array([5.0, -8.0, 2.0])
    _simulate(f, q_true, np.radians([2.0, 1.0, -3.0]), gyro_bias, mag_bias, 3000)
    np.testing.assert_allclose(f.x[3:6], gyro_bias, atol=math.radians(0.01))
    np.testing.assert_allclose(f.x[6:9], mag_bias, atol=0.2)


def test_usque_without_sun_keeps_an_honest_covariance() -> None:
    f = _filter()
    q_true = quat_from_axis_angle([1, 0, 0], 0.3)
    f.reset(q_true, np.r_[np.full(3, 0.001**2), np.full(3, 1e-8), np.full(3, 0.01)])
    p_before = np.trace(f.p[0:3, 0:3])
    _simulate(f, q_true, np.radians([1.0, 1.0, 1.0]), np.zeros(3), np.zeros(3), 600, sun=False)
    # With the magnetometer alone one attitude axis is unobserved at any instant: the
    # attitude covariance must grow rather than collapse.
    assert np.trace(f.p[0:3, 0:3]) > p_before


# --- assembled backend ----------------------------------------------------------------------


def _ctx(**cfg: Any) -> AdcsContext:
    return AdcsContext(
        dt=0.1,
        inertia=Spacecraft().inertia,
        max_dipole=np.array([0.13, 0.13, 0.082]),
        config=AdcsConfig(variant="improved", **cfg),
    )


def _input(q: np.ndarray, t: float = 0.0, **kw: Any) -> AdcsInput:
    a = quat_to_dcm(q)
    base: dict[str, Any] = dict(
        t=t,
        mag_b=a @ MAG_I,
        mag_valid=True,
        gyro_b=np.zeros(3),
        gyro_valid=True,
        sun_b=a @ SUN_I,
        sun_valid=True,
        sun_ref_i=SUN_I,
        mag_ref_i=MAG_I,
    )
    base.update(kw)
    return AdcsInput(**base)


def test_dispatcher_selects_the_variant() -> None:
    adcs = ReferenceAdcs()
    adcs.reset(_ctx())
    assert isinstance(adcs.impl, ImprovedAdcs)


def test_improved_initialises_from_flae_and_reports_its_covariance() -> None:
    adcs = ReferenceAdcs()
    adcs.reset(_ctx())
    q = quat_from_axis_angle([1, 2, 3], 1.0)
    out = adcs.step(_input(q))
    assert out.mode == AdcsMode.ESTIMATE
    assert float(quat_angle(out.q_est, q)) < 1e-6
    assert np.all(np.isfinite(out.p_diag[0:3])) and np.isnan(out.p_diag[3])
    assert np.all(out.p_diag[13:19] > 0)


def test_improved_waits_for_sun_and_magnetometer() -> None:
    adcs = ReferenceAdcs()
    adcs.reset(_ctx())
    out = adcs.step(_input(np.array([0, 0, 0, 1.0]), sun_valid=False))
    assert out.mode == AdcsMode.OFF and out.status & AdcsStatus.NO_ATTITUDE_FIX


def test_improved_detects_a_magnetometer_bias_step() -> None:
    # The body must rotate: with a static geometry a magnetometer bias is indistinguishable
    # from a rotation about the sun direction (both unobservable from one snapshot).
    adcs = ReferenceAdcs()
    adcs.reset(_ctx(sigma_mag_noise=0.1, sigma_sun_noise=1e-3))
    w = np.radians([2.0, 1.0, -3.0])
    q = quat_from_axis_angle([1, 0, 1], 0.5)
    faults = []
    out = None
    for k in range(600):
        bias = np.array([0.0, -2e-5, 0.0]) if k >= 200 else np.zeros(3)
        out = adcs.step(_input(q, t=0.1 * k, gyro_b=w, mag_b=quat_to_dcm(q) @ MAG_I + bias))
        assert out.diag is not None
        faults.append(int(out.diag["fault"]))
        q = quat_mul(rotation_quat(w, 0.1), q)
    assert out is not None
    assert any(faults[200:220])
    np.testing.assert_allclose(out.bias_mag, [0.0, -2e-5, 0.0], atol=2e-7)
    q_last = quat_mul(rotation_quat(-w, 0.1), q)  # the attitude of the last input
    assert float(quat_angle(out.q_est, q_last)) < math.radians(0.2)


def test_bad_input_goes_to_safe_mode() -> None:
    adcs = ReferenceAdcs()
    adcs.reset(_ctx())
    q = np.array([0.0, 0.0, 0.0, 1.0])
    adcs.step(_input(q))
    out = adcs.step(_input(q, t=0.1, gyro_b=np.array([np.inf, 0, 0])))
    assert out.status & AdcsStatus.BAD_INPUT and out.mode == AdcsMode.SAFE


def test_matched_tuning_uses_sensor_noise_with_floors() -> None:
    realistic = matched_improved_config(load_scenario(REPO / "scenarios" / "realistic_1a.toml"))
    assert realistic.variant == "improved"
    assert realistic.sigma_mag_noise == pytest.approx(0.3)
    assert realistic.sigma_gyro_noise == pytest.approx(math.radians(0.03))
    thesis = matched_improved_config(load_scenario(REPO / "scenarios" / "case_1a.toml"))
    assert thesis.sigma_mag_noise == MAG_FLOOR_UT  # 1 pT sensor, floored at 0.1 µT
    assert dataclasses.replace(thesis, variant="thesis").variant == "thesis"


# --- MEKF baseline -----------------------------------------------------------------------------


def test_mekf_transition_matrix_matches_numerical_integration() -> None:
    from adcssim.adcs_ref.mekf import _rotation_matrix
    from adcssim.frames import skew

    w = np.radians([3.0, -5.0, 2.0])
    dt = 0.7
    # δθ̇ = −[ω×]δθ − δβ, integrated with small Euler steps.
    n = 20000
    h = dt / n
    phi11, phi12 = np.eye(3), np.zeros((3, 3))
    for _ in range(n):
        phi12 = phi12 + h * (-skew(w) @ phi12 - np.eye(3))
        phi11 = phi11 + h * (-skew(w) @ phi11)
    exact11, exact12 = _rotation_matrix(w, dt)
    np.testing.assert_allclose(exact11, phi11, atol=1e-4)
    np.testing.assert_allclose(exact12, phi12, atol=1e-4)


def test_mekf_estimates_both_biases() -> None:
    from adcssim.adcs_ref.mekf import Mekf

    f = Mekf(1e-4, 1e-6, 0.1, 1e-3, 1e-3)
    q_true = quat_from_axis_angle([0, 1, 0], 0.4)
    f.reset(q_true, np.r_[np.full(3, 0.01**2), np.full(3, 0.01**2), np.full(3, 50.0**2)])
    gyro_bias = np.radians([0.2, -0.1, 0.3])
    mag_bias = np.array([5.0, -8.0, 2.0])
    _simulate(f, q_true, np.radians([2.0, 1.0, -3.0]), gyro_bias, mag_bias, 3000)  # type: ignore[arg-type]
    np.testing.assert_allclose(f.x[3:6], gyro_bias, atol=math.radians(0.01))
    np.testing.assert_allclose(f.x[6:9], mag_bias, atol=0.2)


def test_dispatcher_selects_the_mekf() -> None:
    from adcssim.adcs_ref.mekf import Mekf

    adcs = ReferenceAdcs()
    adcs.reset(dataclasses.replace(_ctx(), config=AdcsConfig(variant="mekf")))
    assert isinstance(adcs.impl, ImprovedAdcs) and isinstance(adcs.impl.filter, Mekf)
