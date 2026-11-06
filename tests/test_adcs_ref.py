"""Reference ADCS: FLAE, two-stage UKF, controllers and the assembled backend."""

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from adcssim.adcs_ref.adcs import ReferenceAdcs, fault_matrices, process_model
from adcssim.adcs_ref.control import bdot, quaternion_feedback, saturate, torque_to_dipole
from adcssim.adcs_ref.flae import characteristic_coefficients, flae, null_vector
from adcssim.adcs_ref.tsukf import TwoStageUkf, UtWeights, robust_cholesky
from adcssim.backends import AdcsConfig, AdcsContext, AdcsInput, AdcsMode, AdcsStatus
from adcssim.frames import quat_angle, quat_from_axis_angle, quat_normalize, quat_to_dcm
from adcssim.spacecraft import Spacecraft

unit_floats = st.floats(-1.0, 1.0, allow_nan=False)


@st.composite
def quaternions(draw: st.DrawFn) -> np.ndarray:
    q = draw(arrays(np.float64, 4, elements=unit_floats))
    return quat_normalize(q if np.linalg.norm(q) > 1e-3 else np.array([0.0, 0.0, 0.0, 1.0]))


def _pair(rng: np.random.Generator) -> np.ndarray:
    r = rng.normal(size=(2, 3))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


# --- FLAE -------------------------------------------------------------------------------


@given(quaternions())
@settings(max_examples=300)
def test_flae_is_exact_without_noise(q: np.ndarray) -> None:
    r = _pair(np.random.default_rng(1))
    b = r @ quat_to_dcm(q).T
    res = flae(b, r, [0.5, 0.5])
    assert res.ok
    assert math.degrees(float(quat_angle(res.q, q))) < 1e-4
    assert res.lambda_max == pytest.approx(1.0, abs=1e-6)


def test_flae_handles_zero_scalar_part() -> None:
    # A 180° rotation has q4 = 0: elimination that fixes q4 = −1 would divide by zero.
    q = np.array([0.0, 0.6, 0.8, 0.0])
    r = np.eye(3)[:2]
    res = flae(r @ quat_to_dcm(q).T, r, [1.0, 1.0])
    assert res.ok and float(quat_angle(res.q, q)) < 1e-9


def test_flae_sign_follows_previous_estimate() -> None:
    q = quat_from_axis_angle([0.3, 1.0, -0.2], 2.0)
    r = _pair(np.random.default_rng(2))
    b = r @ quat_to_dcm(q).T
    assert flae(b, r, [1, 1], q_prev=-q).q @ (-q) > 0.99
    assert flae(b, r, [1, 1], q_prev=q).q @ q > 0.99


def test_flae_rejects_degenerate_geometry() -> None:
    v = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert not flae(v, v, [1, 1]).ok
    assert not flae(v[:1], v[:1], [1]).ok
    assert not flae(v, v, [0, 0]).ok


def test_flae_weights_favour_the_accurate_sensors() -> None:
    # With two vectors the solution is fixed by the geometry; weights matter with three or
    # more. Two accurate vectors and one noisy one: down-weighting the noisy one helps.
    rng = np.random.default_rng(3)
    q = quat_from_axis_angle([1, 1, 0], 0.7)
    errors: dict[str, list[float]] = {"equal": [], "weighted": []}
    for _ in range(300):
        r = rng.normal(size=(3, 3))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        b = r @ quat_to_dcm(q).T
        b[2] += rng.normal(0, 0.05, 3)
        b[2] /= np.linalg.norm(b[2])
        b[:2] += rng.normal(0, 0.001, (2, 3))
        b[:2] /= np.linalg.norm(b[:2], axis=1, keepdims=True)
        errors["equal"].append(float(quat_angle(flae(b, r, [1, 1, 1]).q, q)))
        errors["weighted"].append(float(quat_angle(flae(b, r, [1, 1, 0.01]).q, q)))
    assert np.mean(errors["weighted"]) < 0.3 * np.mean(errors["equal"])


def test_characteristic_polynomial_matches_numpy() -> None:
    from adcssim.adcs_ref.flae import _w_matrix

    b = np.random.default_rng(4).normal(size=(3, 3))
    w = _w_matrix(b)
    t1, t2, t3 = characteristic_coefficients(b, w)
    np.testing.assert_allclose(np.poly(w), [1.0, 0.0, t1, t2, t3], atol=1e-10)


def test_null_vector() -> None:
    m = np.random.default_rng(5).normal(size=(4, 4))
    m[:, 3] = m[:, :3] @ np.array([1.0, -2.0, 0.5])  # rank 3
    v = null_vector(m)
    assert v is not None
    np.testing.assert_allclose(m @ v, 0.0, atol=1e-10)
    assert null_vector(np.zeros((4, 4))) is None


# --- unscented transform and two-stage UKF -----------------------------------------------


def test_ut_weights_sum_to_one() -> None:
    w = UtWeights.thesis(alpha=1.0, n=13)
    assert w.wm0 + 2 * 13 * w.wi == pytest.approx(1.0)
    assert w.xi == pytest.approx(math.sqrt(3.0))


def test_robust_cholesky_recovers_semidefinite_matrix() -> None:
    v = np.array([1.0, 2.0, 3.0])
    p = np.outer(v, v)  # rank 1
    lower = robust_cholesky(p)
    assert lower is not None
    np.testing.assert_allclose(lower @ lower.T, p, atol=1e-6)


def test_tsukf_separates_state_and_sensor_bias() -> None:
    """One constant state seen by two sensors, the second with an unknown bias: the filter
    must find both the state and the bias (they are observable from the difference)."""
    fa = np.zeros((1, 1))
    fs = np.array([[0.0], [1.0]])
    filt = TwoStageUkf(fa, fs, np.full(1, 1e-10), np.full(1, 1e-10), np.full(2, 1e-4), alpha=1.0)
    filt.reset(np.zeros(1), np.full(1, 1.0), np.zeros(1), np.full(1, 1.0))
    rng = np.random.default_rng(6)
    for _ in range(300):
        y = np.array([0.3, 0.3 + 0.5]) + rng.normal(0, 1e-2, 2)
        filt.step(lambda xs: xs, y, np.array([True, True]), h=lambda xs: np.hstack([xs, xs]))
    assert filt.state is not None
    assert filt.state.x[0] == pytest.approx(0.3, abs=0.01)
    assert filt.state.b[0] == pytest.approx(0.5, abs=0.01)


def test_tsukf_ignores_unavailable_measurements() -> None:
    fa, fs = np.zeros((1, 1)), np.array([[0.0], [1.0]])
    filt = TwoStageUkf(fa, fs, np.full(1, 1e-10), np.full(1, 1e-10), np.full(2, 1e-4), alpha=1.0)
    filt.reset(np.array([0.3]), np.full(1, 1e-4), np.zeros(1), np.full(1, 1e-4))
    for _ in range(20):  # second sensor reports garbage but is flagged unavailable
        filt.step(
            lambda xs: xs,
            np.array([0.3, 1e6]),
            np.array([True, False]),
            h=lambda xs: np.hstack([xs, xs]),
        )
    assert filt.state is not None
    assert filt.state.x[0] == pytest.approx(0.3, abs=1e-3)
    assert abs(filt.state.b[0]) < 1e-3


def test_tsukf_estimates_a_sensor_bias_with_the_thesis_model() -> None:
    """The thesis structure: the gyro bias is observable because the attitude measurement
    constrains ω through the kinematics. Run the full reference ADCS on synthetic data with a
    constant gyro bias and check the estimate."""
    adcs = ReferenceAdcs()
    sc = Spacecraft()
    adcs.reset(AdcsContext(dt=0.1, inertia=sc.inertia, max_dipole=np.full(3, 0.1)))
    bias = np.radians([0.0, 0.0, 2.0])
    q = quat_from_axis_angle([1, 0, 0], 0.2)
    w = np.radians([1.0, -0.5, 0.8])
    sun_i, mag_i = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2e-5, 3e-5])
    rng = np.random.default_rng(7)
    out = None
    for k in range(1200):
        # Constant-rate truth (spherical inertia approximation is fine over a few minutes).
        q = quat_normalize(q + 0.05 * np.concatenate([q[3] * w + np.cross(q[:3], w), [-q[:3] @ w]]))
        a = quat_to_dcm(q)
        out = adcs.step(
            AdcsInput(
                t=k * 0.1,
                mag_b=a @ mag_i + rng.normal(0, 1e-7, 3),
                mag_valid=True,
                gyro_b=w + bias + rng.normal(0, 1e-4, 3),
                gyro_valid=True,
                sun_b=a @ sun_i,
                sun_valid=True,
                sun_ref_i=sun_i,
                mag_ref_i=mag_i,
            )
        )
    assert out is not None
    np.testing.assert_allclose(out.bias_gyro, bias, atol=math.radians(0.2))


# --- controllers -------------------------------------------------------------------------


def test_bdot_opposes_the_field_change() -> None:
    np.testing.assert_allclose(bdot(np.array([1e-6, 0, 0]), 4e4), [-0.04, 0, 0])


def test_quaternion_feedback_takes_the_short_way() -> None:
    target = np.array([0.0, 0.0, 0.0, 1.0])
    q = quat_from_axis_angle([0, 0, 1], math.radians(30))
    tau = quaternion_feedback(q, target, np.zeros(3), kp=1.0, kd=0.0)
    assert tau[2] < 0  # rotate back about −z
    # −q is the same attitude: same torque.
    np.testing.assert_allclose(quaternion_feedback(-q, target, np.zeros(3), 1.0, 0.0), tau)
    # Damping opposes the rate.
    tau_d = quaternion_feedback(target, target, np.array([0.1, 0, 0]), kp=0.0, kd=2.0)
    np.testing.assert_allclose(tau_d, [-0.2, 0, 0])


def test_torque_to_dipole_produces_the_perpendicular_torque() -> None:
    b = np.array([1e-5, 2e-5, -3e-5])
    tau = np.array([1e-7, -2e-7, 0.5e-7])
    m = torque_to_dipole(tau, b)
    tau_perp = tau - (tau @ b) / (b @ b) * b
    np.testing.assert_allclose(np.cross(m, b), tau_perp, atol=1e-20)


def test_saturation_keeps_direction() -> None:
    m, sat = saturate(np.array([0.26, 0.13, 0.0]), np.array([0.13, 0.13, 0.082]))
    assert sat
    np.testing.assert_allclose(m, [0.13, 0.065, 0.0])
    m2, sat2 = saturate(np.array([0.01, 0.0, 0.0]), np.full(3, 0.1))
    assert not sat2 and m2[0] == 0.01


# --- process model -------------------------------------------------------------------------


def test_process_model_matches_rigid_body_kinematics() -> None:
    sc = Spacecraft()
    q0 = quat_from_axis_angle([0.2, 0.4, 1.0], 0.5)
    w0 = np.radians([2.0, -1.0, 3.0])
    rot = quat_to_dcm(q0)
    x0 = np.concatenate([q0, rot @ [1.0, 0, 0], rot @ [0, 30.0, 0], w0])
    x1 = process_model(
        x0[None, :],
        1.0,
        0.1,
        rot,
        np.array([1.0, 0, 0]),
        np.array([0, 30.0, 0]),
        np.zeros(3),
        sc.inertia,
        np.linalg.inv(sc.inertia),
    )[0]
    from adcssim.dynamics import rk4_step

    q_ref, w_ref = q0, w0
    for _ in range(10):
        q_ref, w_ref = rk4_step(q_ref, w_ref, 0.1, sc.inertia, lambda q, w: np.zeros(3))
    assert float(quat_angle(quat_normalize(x1[:4]), q_ref)) < 1e-9
    np.testing.assert_allclose(x1[10:], w_ref, atol=1e-12)


def test_fault_matrices_map_biases_onto_b_and_omega() -> None:
    fa, fs = fault_matrices()
    assert not fa.any()
    np.testing.assert_array_equal(fs[7:10, 0:3], np.eye(3))
    np.testing.assert_array_equal(fs[10:13, 3:6], np.eye(3))
    assert fs[:7].sum() == 0


# --- assembled backend ---------------------------------------------------------------------


def _input(**kw: Any) -> AdcsInput:
    base: dict[str, Any] = dict(
        t=0.0,
        mag_b=np.array([1e-5, 2e-5, 3e-5]),
        mag_valid=True,
        gyro_b=np.zeros(3),
        gyro_valid=True,
        sun_b=np.array([1.0, 0.0, 0.0]),
        sun_valid=True,
        sun_ref_i=np.array([1.0, 0.0, 0.0]),
        mag_ref_i=np.array([1e-5, 2e-5, 3e-5]),
    )
    base.update(kw)
    return AdcsInput(**base)


def _adcs(**cfg: Any) -> ReferenceAdcs:
    adcs = ReferenceAdcs()
    adcs.reset(
        AdcsContext(
            dt=0.1,
            inertia=Spacecraft().inertia,
            max_dipole=np.array([0.13, 0.13, 0.082]),
            config=AdcsConfig(**cfg),
        )
    )
    return adcs


def test_backend_initialises_from_the_first_fix() -> None:
    adcs = _adcs()
    out = adcs.step(_input())
    assert out.mode == AdcsMode.ESTIMATE
    np.testing.assert_allclose(out.q_est, [0, 0, 0, 1], atol=1e-6)
    assert out.p_diag.shape == (19,) and np.all(out.p_diag > 0)


def test_backend_waits_for_a_fix() -> None:
    adcs = _adcs()
    out = adcs.step(_input(sun_valid=False))
    assert out.mode == AdcsMode.OFF
    assert out.status & AdcsStatus.NO_ATTITUDE_FIX
    assert np.all(np.isnan(out.q_est))


def test_bad_input_goes_to_safe_mode() -> None:
    adcs = _adcs()
    adcs.step(_input())
    out = adcs.step(_input(t=0.1, gyro_b=np.array([np.nan, 0, 0])))
    assert out.status & AdcsStatus.BAD_INPUT
    assert out.mode == AdcsMode.SAFE
    np.testing.assert_array_equal(out.m_cmd, 0.0)


def test_control_disabled_never_commands_the_coils() -> None:
    adcs = _adcs()
    for k in range(20):
        out = adcs.step(_input(t=0.1 * k, gyro_b=np.radians([10.0, 0, 0])))
        np.testing.assert_array_equal(out.m_cmd, 0.0)
        assert out.mode == AdcsMode.ESTIMATE


def test_fast_rotation_triggers_detumbling() -> None:
    adcs = _adcs(control_enabled=True)
    modes = []
    for k in range(30):
        out = adcs.step(_input(t=0.1 * k, gyro_b=np.radians([10.0, 0, 0])))
        modes.append(out.mode)
    assert AdcsMode.DETUMBLE in modes


def test_not_initialised() -> None:
    out = ReferenceAdcs().step(_input())
    assert out.status & AdcsStatus.NOT_INITIALISED


# --- diagnostics and variants ----------------------------------------------------------------


def test_diagnostics_report_innovation_statistics() -> None:
    adcs = _adcs()
    adcs.step(_input())
    out = adcs.step(_input(t=0.1))
    assert out.diag is not None
    assert out.diag["innov"].shape == (13,)
    assert out.diag["nis"].shape == (4,) and np.all(out.diag["nis"] >= 0.0)
    assert out.diag["bias_gain"].shape == (6,)
    np.testing.assert_allclose(np.linalg.norm(out.diag["q_flae"]), 1.0)


def test_diagnostics_mark_missing_measurements() -> None:
    adcs = _adcs()
    adcs.step(_input())
    out = adcs.step(_input(t=0.1, sun_valid=False))
    assert out.diag is not None
    assert np.all(np.isnan(out.diag["innov"][4:7]))  # sun rows unavailable
    assert np.isnan(out.diag["nis"][1])


def test_only_known_variants_are_accepted() -> None:
    assert AdcsConfig().variant == "thesis"
    with pytest.raises(ValueError, match="variant"):
        AdcsConfig(variant="magic")
