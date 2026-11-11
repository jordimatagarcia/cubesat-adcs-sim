"""Regression tests for the defects found in the 2026-11 code review (docs/adcs.md,
"Review fixes"). Each test reproduces the original failure and checks the fixed behaviour.
The C and Rust libraries implement the same fixes; tests/test_native.py runs them on the
golden vectors and checks they accept and reject the same configurations."""

import math
from dataclasses import replace

import numpy as np
import pytest

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.adcs_ref.control import ModeController, saturate
from adcssim.adcs_ref.flae import flae
from adcssim.adcs_ref.usque import rotation_quat
from adcssim.backends import AdcsConfig, AdcsContext, AdcsInput, AdcsMode, AdcsStatus
from adcssim.frames import quat_angle, quat_from_axis_angle, quat_mul, quat_to_dcm
from adcssim.spacecraft import Spacecraft

SUN_I = np.array([0.6, 0.8, 0.0])
INERTIA = Spacecraft().inertia
MAX_DIPOLE = np.array([0.13, 0.13, 0.082])


def mag_i(t: float) -> np.ndarray:
    a = 2 * math.pi * t / 5400.0
    return 3e-5 * np.array([math.cos(a), 0.5 * math.sin(a), 0.8 * math.sin(a)])


def make(variant: str = "mekf", **cfg: object) -> ReferenceAdcs:
    adcs = ReferenceAdcs()
    adcs.reset(
        AdcsContext(
            dt=0.1,
            inertia=INERTIA,
            max_dipole=MAX_DIPOLE,
            config=AdcsConfig(variant=variant, **cfg),  # type: ignore[arg-type]
        )
    )
    return adcs


def reading(
    t: float, q: np.ndarray, w: np.ndarray, rng: np.random.Generator, **kw: object
) -> AdcsInput:
    a = quat_to_dcm(q)
    d: dict[str, object] = dict(
        t=t,
        mag_b=a @ mag_i(t) + 1e-7 * rng.normal(size=3),
        mag_valid=True,
        gyro_b=w + 1e-4 * rng.normal(size=3),
        gyro_valid=True,
        sun_b=a @ SUN_I + 1e-3 * rng.normal(size=3),
        sun_valid=True,
        sun_ref_i=SUN_I,
        mag_ref_i=mag_i(t),
    )
    d.update(kw)
    return AdcsInput(**d)  # type: ignore[arg-type]


def torque_free(
    q: np.ndarray, w: np.ndarray, dt: float = 0.1, n: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    h = dt / n
    inv = np.linalg.inv(INERTIA)
    for _ in range(n):
        wd = inv @ -np.cross(w, INERTIA @ w)
        wm = w + 0.5 * h * wd
        q = quat_mul(rotation_quat(wm, h), q)
        q /= np.linalg.norm(q)
        w = w + h * wd
    return q, w


@pytest.mark.parametrize("variant", ["mekf", "improved"])
def test_gyro_outage_in_eclipse_stays_consistent(variant: str) -> None:
    """Was: 60 s without the gyro while tumbling in eclipse → 180° error with a reported
    1σ of 0.12° (1466σ), and a corrupted gyro-bias estimate afterwards."""
    adcs = make(variant)
    rng = np.random.default_rng(3)
    q, w = quat_from_axis_angle([1, 2, 3], 0.5), np.radians([3.0, -6.0, 4.0])
    worst_ratio, rate_err_after, flags = 0.0, 0.0, 0
    for k in range(1200):
        kw: dict[str, object] = {"sun_valid": k < 290}
        if 300 <= k < 900:
            kw.update(gyro_valid=False, gyro_b=np.full(3, np.nan))
        out = adcs.step(reading(k * 0.1, q, w, rng, **kw))
        if k >= 300:
            err = float(quat_angle(out.q_est, q))
            worst_ratio = max(worst_ratio, err / math.sqrt(max(out.p_diag[0:3])))
            if 300 <= k < 900:
                flags |= out.status
            if k >= 1000:
                rate_err_after = max(rate_err_after, float(np.linalg.norm(out.w_est - w)))
        q, w = torque_free(q, w)
    assert flags & AdcsStatus.RATE_PROPAGATED
    assert worst_ratio < 2.0
    assert math.degrees(rate_err_after) < 0.1


def _sun_run(scale: float) -> tuple[np.ndarray, int]:
    adcs = make("mekf")
    rng = np.random.default_rng(0)
    q, w = quat_from_axis_angle([1, -1, 2], 0.3), np.radians([0.5, -0.3, 0.2])
    errors, faults = [], 0
    for k in range(600):
        out = adcs.step(reading(k * 0.1, q, w, rng, sun_ref_i=SUN_I * scale))
        errors.append(math.degrees(float(quat_angle(out.q_est, q))))
        faults += bool(out.status & AdcsStatus.BIAS_FAULT)
        q, w = torque_free(q, w)
    return np.array(errors), faults


@pytest.mark.parametrize("scale", [0.983, 0.6, 1.9])
def test_sun_vectors_are_normalised(scale: float) -> None:
    """Was: a Sun direction of length 0.983 (an ephemeris in AU) gave the MEKF 18° error
    and false bias faults; lengths are accepted from 0.5 to 2. Now the result is the same
    as with a unit vector."""
    errors, faults = _sun_run(scale)
    baseline, _ = _sun_run(1.0)
    assert faults == 0
    assert np.max(np.abs(errors - baseline)) < 1e-6


@pytest.mark.parametrize("scale", [1.0, 1.5, 1.8, 2.0, 0.5])
def test_flae_finds_the_largest_eigenvalue_for_long_vectors(scale: float) -> None:
    """Was: from length ~1.5–1.8, Newton from λ = 1 converged to a smaller root and FLAE
    returned a 180°-wrong attitude marked as success."""
    q = quat_from_axis_angle([1, -2, 0.5], 1.2)
    a = quat_to_dcm(q)
    s = np.array([0.6, 0.8, 0.0])
    b = np.array([0.3, -0.4, 0.866]) / np.linalg.norm([0.3, -0.4, 0.866])
    res = flae(np.vstack([scale * (a @ s), a @ b]), np.vstack([s, b]), [1, 1], tolerance=1e-12)
    assert res.ok
    assert math.degrees(float(quat_angle(res.q, q))) < 1e-6


def test_bdot_keeps_working_without_the_field_model() -> None:
    """Was: with the on-board field model invalid, B-dot got B = 0: a saturated spurious
    pulse, then no detumbling for the whole outage."""
    adcs = make("mekf", control_enabled=True)
    rng = np.random.default_rng(1)
    q, w = quat_from_axis_angle([1, 0, 0], 0.2), np.radians([6.0, -4.0, 3.0])
    commands = []
    for k in range(80):
        ref = np.zeros(3) if 40 <= k < 60 else mag_i(k * 0.1)
        out = adcs.step(reading(k * 0.1, q, w, rng, mag_ref_i=ref))
        if k >= 30:
            assert out.mode == AdcsMode.DETUMBLE
            commands.append(out.m_cmd)
        q, w = torque_free(q, w)
    m = np.array(commands)
    # The command during the outage continues smoothly from before it.
    assert np.all(np.linalg.norm(m[10:30], axis=1) > 0.1 * np.linalg.norm(m[:10], axis=1).min())
    jump = np.max(np.abs(np.diff(m, axis=0)))
    assert jump < 0.5 * float(np.max(MAX_DIPOLE))


def test_bdot_uses_the_rate_not_the_noisy_field_difference() -> None:
    """Was: the MEKF's B-dot differentiated the raw magnetometer; at 0.3 µT noise 90% of
    the commands saturated and many pointed the wrong way. Ḃ = −ω̂ × B instead."""
    adcs = make("mekf", control_enabled=True)
    rng = np.random.default_rng(2)
    q, w = quat_from_axis_angle([0, 1, 0], 0.4), np.radians([5.0, -6.0, 3.0])
    cosines = []
    for k in range(200):
        a = quat_to_dcm(q)
        noisy = a @ mag_i(k * 0.1) + 3e-7 * rng.normal(size=3)
        out = adcs.step(reading(k * 0.1, q, w, rng, mag_b=noisy))
        if k > 50 and out.mode == AdcsMode.DETUMBLE:
            ideal = 4e4 * np.cross(w, a @ mag_i(k * 0.1))  # −k Ḃ, Ḃ = −ω × B
            cosines.append(
                float(out.m_cmd @ ideal / np.linalg.norm(out.m_cmd) / np.linalg.norm(ideal))
            )
        q, w = torque_free(q, w)
    assert len(cosines) > 100
    assert np.median(cosines) > 0.99


def test_safe_resumes_the_interrupted_mode() -> None:
    """Was: one bad step aborted detumbling inside the hysteresis band, and restarted the
    settle time from POINT."""
    cfg = AdcsConfig(control_enabled=True)
    q, b = np.array([0.0, 0, 0, 1]), np.array([2e-5, 0, 3e-5])
    ok, bad = AdcsStatus.OK, AdcsStatus.BAD_INPUT

    c = ModeController()
    c.update(cfg, 0.1, MAX_DIPOLE, q, np.radians([5.0, 0, 0]), b, ok)  # → DETUMBLE
    slow = np.radians([2.0, 0, 0])  # between the exit (1°/s) and enter (4°/s) rates
    assert c.update(cfg, 0.1, MAX_DIPOLE, q, slow, b, ok)[0] == AdcsMode.DETUMBLE
    assert c.update(cfg, 0.1, MAX_DIPOLE, q, slow, b, bad)[0] == AdcsMode.SAFE
    assert c.update(cfg, 0.1, MAX_DIPOLE, q, slow, b, ok)[0] == AdcsMode.DETUMBLE

    c = ModeController()
    still = np.zeros(3)
    short = replace(cfg, settle_time=1.0)
    modes = [c.update(short, 0.1, MAX_DIPOLE, q, still, b, ok)[0] for _ in range(15)]
    assert modes[-1] == AdcsMode.POINT
    assert c.update(short, 0.1, MAX_DIPOLE, q, still, b, bad)[0] == AdcsMode.SAFE
    assert c.update(short, 0.1, MAX_DIPOLE, q, still, b, ok)[0] == AdcsMode.POINT


def test_non_finite_dipole_command_turns_the_coils_off() -> None:
    m, saturated = saturate(np.array([np.nan, 0.01, 0.0]), MAX_DIPOLE)
    assert np.array_equal(m, np.zeros(3)) and not saturated


BAD_CONFIGS = [
    ("dt", {}, {"dt": 0.0}),
    ("dt", {}, {"dt": 20.0}),
    ("inertia", {}, {"inertia": np.diag([0.04, 0.04, -0.007])}),
    ("inertia", {}, {"inertia": np.array([[0.04, 0.01, 0], [0, 0.04, 0], [0, 0, 0.007]])}),
    ("max_dipole", {}, {"max_dipole": np.array([0.13, 0.0, 0.08])}),
    ("process_dt", {"process_dt": 1e-6}, {}),
    ("ukf_alpha", {"ukf_alpha": 1e-6}, {}),
    ("flae_max_iterations", {"flae_max_iterations": 0}, {}),
    ("kp", {"kp": 1e306}, {}),
    ("fault_steps", {"variant": "mekf", "fault_steps": 0}, {}),
    ("sigma_outage_accel", {"variant": "mekf", "sigma_outage_accel": -1.0}, {}),
]


@pytest.mark.parametrize(("rule", "cfg", "ctx"), BAD_CONFIGS, ids=[r for r, *_ in BAD_CONFIGS])
def test_invalid_configurations_are_rejected(
    rule: str, cfg: dict[str, object], ctx: dict[str, object]
) -> None:
    """Was: the C and Rust libraries accepted fault_steps = 0 (a fault every step),
    flae_max_iterations = 0 (never initialises), dt / process_dt = 1e12 (undefined behaviour
    in C, a hang everywhere) and more; Python accepted negative gains and dt = 0."""
    args = {"dt": 0.1, "inertia": INERTIA, "max_dipole": MAX_DIPOLE} | ctx
    with pytest.raises(ValueError, match=rule):
        try:
            config = AdcsConfig(**cfg)  # type: ignore[arg-type]
        except ValueError as e:  # some rules already fire when the config is built
            raise ValueError(f"{rule}: {e}") from e
        config.validate(args["dt"], args["inertia"], args["max_dipole"])  # type: ignore[arg-type]
