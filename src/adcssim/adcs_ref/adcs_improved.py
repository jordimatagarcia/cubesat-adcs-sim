"""The improved ADCS variant: error-state UKF (USQUE) with vector measurements, gyro and
magnetometer bias states, and innovation-based fault detection. Variant ``mekf`` runs the
same model with a multiplicative EKF (``mekf.py``) instead.

Designed from the diagnosis of the thesis variant (docs/results/diagnosis.md):

- The sun and magnetometer vectors are measured against the attitude directly, so the
  magnetometer constrains the attitude in eclipse and its bias is observable; FLAE is used
  only to initialise.
- The attitude is a 3-parameter error around a reference quaternion (no 4-state
  quaternion inside the UKF).
- The gyro bias is a slow random walk, so it no longer absorbs gyro noise. Sudden bias
  faults (the thesis test cases) are handled by a detector instead: when the sun or
  magnetometer innovation stays above ``fault_nis_threshold`` for ``fault_steps`` steps in
  sunlight, the attitude and both bias uncertainties are inflated and the measurements
  re-estimate them. The logged fault flag says which innovation was larger (1 sun → likely
  gyro, 2 magnetometer).
- The gyro is the rate input: no dynamics model, so unmodelled torques (control, duty
  cycle, disturbances) don't enter the estimate. While the gyro is unavailable the rate is
  propagated with Euler's torque-free equations instead, the attitude uncertainty grows
  with what they leave out (``sigma_outage_accel``), the gyro-bias estimate is frozen (it
  cannot be observed without the gyro and would otherwise absorb the propagation error),
  and the status reports ``RATE_PROPAGATED``.
- B-dot uses Ḃ = −ω̂ × B with the bias-corrected rate, not the difference of successive
  magnetometer readings, which is mostly noise at 10 Hz.
"""

import numpy as np
from numpy.typing import NDArray

from adcssim.adcs_ref.control import ModeController
from adcssim.adcs_ref.flae import flae
from adcssim.adcs_ref.limits import field_plausible, gyro_plausible, unit, unit_plausible
from adcssim.adcs_ref.mekf import Mekf
from adcssim.adcs_ref.usque import ATT, GYRO, MAG, Usque
from adcssim.backends.base import AdcsContext, AdcsInput, AdcsMode, AdcsOutput, AdcsStatus
from adcssim.backends.config import AdcsConfig
from adcssim.frames import quat_rotate

FloatArray = NDArray[np.float64]
UT = 1e6

FAULT_NONE, FAULT_GYRO, FAULT_MAG = 0, 1, 2


def _euler_rate(w: FloatArray, inertia: FloatArray, inertia_inv: FloatArray) -> FloatArray:
    """Torque-free Euler equations, ω̇ = I⁻¹ (−ω × Iω)."""
    return np.asarray(inertia_inv @ np.cross(-w, inertia @ w), dtype=np.float64)


def propagate_rate(
    w: FloatArray, dt: float, inertia: FloatArray, inertia_inv: FloatArray
) -> FloatArray:
    """Body rate after ``dt`` without torques (one RK4 step)."""
    k1 = _euler_rate(w, inertia, inertia_inv)
    k2 = _euler_rate(w + 0.5 * dt * k1, inertia, inertia_inv)
    k3 = _euler_rate(w + 0.5 * dt * k2, inertia, inertia_inv)
    k4 = _euler_rate(w + dt * k3, inertia, inertia_inv)
    return np.asarray(w + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, dtype=np.float64)


OUTAGE_MAX_ATTITUDE_SIGMA = 0.25
"""rad. The outage growth stops once an attitude variance reaches this σ²: beyond ~15° the
attitude is effectively unknown, and larger values only make the filters' linearisation
(MEKF) or sigma points (UKF, which spread past 180°) fail."""


def outage_variance(sigma_accel: float, t0: float, t1: float) -> float:
    """Attitude variance added from ``t0`` to ``t1`` seconds into a gyro outage. A rate
    error growing as σ_α t gives an attitude error σ_α t²/2, so the variance grows as
    (σ_α²/4) t⁴."""
    return 0.25 * sigma_accel * sigma_accel * (t1 * t1 * t1 * t1 - t0 * t0 * t0 * t0)


class ImprovedAdcs:
    name = "python-ref"

    def __init__(self) -> None:
        self.ctx: AdcsContext | None = None
        self.filter: Usque | Mekf | None = None

    @property
    def config(self) -> AdcsConfig:
        assert self.ctx is not None
        return self.ctx.config

    def reset(self, ctx: AdcsContext) -> None:
        cfg = ctx.config
        cfg.validate(ctx.dt, ctx.inertia, ctx.max_dipole)
        self.ctx = ctx
        self.inertia = np.asarray(ctx.inertia, dtype=np.float64).reshape(3, 3)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.outage_time = 0.0
        noise = (
            cfg.sigma_gyro_noise,
            cfg.sigma_gyro_bias_walk,
            cfg.sigma_mag_noise,
            cfg.sigma_mag_bias_walk,
            cfg.sigma_sun_noise,
        )
        self.filter = Mekf(*noise) if cfg.variant == "mekf" else Usque(cfg.usque_lambda, *noise)
        self.control = ModeController()
        self.initialised = False
        self.gyro_prev = np.zeros(3)
        self.suspect_steps = 0
        self.fault = FAULT_NONE

    def step(self, inp: AdcsInput) -> AdcsOutput:
        if self.ctx is None or self.filter is None:
            return AdcsOutput(status=int(AdcsStatus.NOT_INITIALISED))
        cfg = self.config
        status = AdcsStatus.OK
        self.fault = FAULT_NONE
        self.filter.last_update = None

        # A reading is used if it is finite and physically plausible (limits.py).
        mag_ref_ok = field_plausible(inp.mag_ref_i)
        sun_ref_ok = unit_plausible(inp.sun_ref_i)
        mag_plausible = field_plausible(inp.mag_b)
        sun_plausible = unit_plausible(inp.sun_b)
        mag_ok = inp.mag_valid and mag_plausible and mag_ref_ok
        sun_ok = inp.sun_valid and sun_plausible and sun_ref_ok
        gyro_ok = inp.gyro_valid and gyro_plausible(inp.gyro_b)
        if (
            (inp.mag_valid and not mag_plausible)
            or (inp.sun_valid and not sun_plausible)
            or (inp.gyro_valid and not gyro_ok)
        ):
            status |= AdcsStatus.BAD_INPUT
        # The magnetometer reading itself (for B-dot) needs no reference model.
        mag_reading_ok = inp.mag_valid and mag_plausible
        mag_ut = np.asarray(inp.mag_b, dtype=np.float64) * UT if mag_ok else np.zeros(3)
        mag_meas_ut = (
            np.asarray(inp.mag_b, dtype=np.float64) * UT if mag_reading_ok else np.zeros(3)
        )
        mag_ref_ut = np.asarray(inp.mag_ref_i, dtype=np.float64) * UT if mag_ref_ok else np.zeros(3)
        # Directions are normalised: the plausibility check accepts lengths 0.5–2.
        sun_b = unit(inp.sun_b) if sun_ok else np.zeros(3)
        sun_ref = unit(inp.sun_ref_i) if sun_ref_ok else np.zeros(3)
        gyro = np.asarray(inp.gyro_b, dtype=np.float64) if gyro_ok else self.gyro_prev

        if not (sun_ok and mag_ok):
            status |= AdcsStatus.NO_ATTITUDE_FIX

        if not self.initialised:
            if not (sun_ok and mag_ok and gyro_ok):
                return AdcsOutput(mode=AdcsMode.OFF, status=int(status), diag=self._diag(None))
            res = flae(
                np.vstack([sun_b, mag_ut / np.linalg.norm(mag_ut)]),
                np.vstack([sun_ref, mag_ref_ut / np.linalg.norm(mag_ref_ut)]),
                [cfg.flae_weight_sun, cfg.flae_weight_mag],
                tolerance=cfg.flae_tolerance,
                max_iterations=cfg.flae_max_iterations,
            )
            if not res.ok:
                status |= AdcsStatus.FLAE_FAILED
                return AdcsOutput(mode=AdcsMode.OFF, status=int(status), diag=self._diag(None))
            p0 = np.r_[
                np.full(3, cfg.p0_attitude**2),
                np.full(3, cfg.p0_gyro_bias**2),
                np.full(3, cfg.p0_mag_bias**2),
            ]
            self.filter.reset(res.q, p0)
            self.initialised = True
            self.gyro_prev = gyro
            return self._finish(inp, status, gyro, mag_meas_ut, mag_reading_ok, res.q)

        # Propagate with the mean of the previous and current rate readings: the body rate
        # changes within a step (Euler coupling while tumbling), and holding the previous
        # reading constant leaves an O(ω̇ dt²) attitude error per step.
        dt = self.ctx.dt
        outage_var = 0.0
        if gyro_ok:
            self.outage_time = 0.0
        else:
            bias = self.filter.x[GYRO]
            gyro = propagate_rate(self.gyro_prev - bias, dt, self.inertia, self.inertia_inv) + bias
            t0 = self.outage_time
            self.outage_time = t0 + dt
            outage_var = outage_variance(cfg.sigma_outage_accel, t0, self.outage_time)
            p = self.filter.p
            largest = max(p[0, 0], p[1, 1], p[2, 2])
            cap = OUTAGE_MAX_ATTITUDE_SIGMA * OUTAGE_MAX_ATTITUDE_SIGMA
            outage_var = min(outage_var, max(0.0, cap - largest))
            status |= AdcsStatus.RATE_PROPAGATED
        rate = 0.5 * (self.gyro_prev + gyro)
        ok = self.filter.step(
            rate,
            dt,
            sun_b,
            sun_ref,
            sun_ok,
            mag_ut,
            mag_ref_ut,
            mag_ok,
            extra_attitude_var=outage_var,
            freeze_gyro_bias=not gyro_ok,
        )
        if not ok:
            status |= AdcsStatus.COVARIANCE_RESET
        self._detect_faults(sun_ok)
        if self.fault != FAULT_NONE:
            status |= AdcsStatus.BIAS_FAULT
        self.gyro_prev = gyro
        return self._finish(inp, status, gyro, mag_meas_ut, mag_reading_ok, None)

    # ------------------------------------------------------------------------------------
    def _nis(self) -> tuple[float, float]:
        upd = self.filter.last_update if self.filter is not None else None
        out = []
        for rows in (slice(0, 3), slice(3, 6)):
            if upd is None or not upd.available[rows].all():
                out.append(float("nan"))
                continue
            e = upd.innovation[rows]
            out.append(float(e @ np.linalg.solve(upd.innovation_cov[rows, rows], e)))
        return out[0], out[1]

    def _detect_faults(self, sun_ok: bool) -> None:
        """Declare a bias fault when the sun or magnetometer innovation stays too large.

        Which sensor is at fault isn't guessed: a magnetometer-bias step first pulls the
        attitude, which then also shows up in the sun innovation. Instead the attitude and
        both bias uncertainties are inflated and the measurements sort it out (the sun
        sensor has no bias, so it pins the attitude). Only in sunlight: in eclipse there is
        no unbiased reference to arbitrate."""
        assert self.filter is not None
        cfg = self.config
        nis_sun, nis_mag = self._nis()
        thr = cfg.fault_nis_threshold
        suspect = sun_ok and (nis_sun > thr or nis_mag > thr)
        self.suspect_steps = self.suspect_steps + 1 if suspect else 0
        if self.suspect_steps >= cfg.fault_steps:
            self.filter.inflate(
                attitude=cfg.fault_attitude_sigma,
                gyro_bias=cfg.fault_gyro_bias_sigma,
                mag_bias=cfg.fault_mag_bias_sigma,
            )
            self.suspect_steps = 0
            self.fault = FAULT_GYRO if nis_sun > nis_mag else FAULT_MAG

    def _finish(
        self,
        inp: AdcsInput,
        status: AdcsStatus,
        gyro: FloatArray,
        mag_ut: FloatArray,
        mag_ok: bool,
        q_flae: FloatArray | None,
    ) -> AdcsOutput:
        """Control and output. ``mag_ut``/``mag_ok``: the magnetometer reading, usable for
        B-dot whether or not the field model is available."""
        assert self.ctx is not None and self.filter is not None
        f = self.filter
        q = f.q_ref
        w = gyro - f.x[GYRO]
        if mag_ok:
            b_body = (mag_ut - f.x[MAG]) / UT
        elif field_plausible(inp.mag_ref_i):
            b_body = quat_rotate(q, np.asarray(inp.mag_ref_i, dtype=np.float64))
        else:
            b_body = np.zeros(3)
        mode, m, status = self.control.update(
            self.config,
            self.ctx.dt,
            np.asarray(self.ctx.max_dipole, dtype=np.float64),
            q,
            w,
            b_body,
            status,
            b_dot=-np.cross(w, b_body),  # a fixed inertial field seen from the rotating body
        )
        cfg = self.config
        p = f.p
        p_diag = np.full(19, np.nan)
        p_diag[0:3] = np.diag(p)[ATT]  # attitude error variance per axis (rad²)
        rate_outage = cfg.sigma_outage_accel * self.outage_time
        # rate = gyro − bias (+ the propagation uncertainty during a gyro outage)
        p_diag[10:13] = cfg.sigma_gyro_noise**2 + np.diag(p)[GYRO] + rate_outage * rate_outage
        p_diag[13:16] = np.diag(p)[MAG]
        p_diag[16:19] = np.diag(p)[GYRO]
        return AdcsOutput(
            q_est=q.copy(),
            w_est=w,
            bias_mag=f.x[MAG] / UT,
            bias_gyro=f.x[GYRO].copy(),
            p_diag=p_diag,
            m_cmd=m,
            mode=mode,
            status=int(status),
            diag=self._diag(q_flae),
        )

    def _diag(self, q_flae: FloatArray | None) -> dict[str, FloatArray]:
        d: dict[str, FloatArray] = {
            "q_flae": q_flae if q_flae is not None else np.full(4, np.nan),
            "flae_lambda": np.array(np.nan),
            "innov": np.full(13, np.nan),
            "nis": np.full(4, np.nan),
            "bias_gain": np.full(6, np.nan),
            "fault": np.array(float(self.fault)),
        }
        upd = self.filter.last_update if self.filter is not None else None
        if upd is not None:
            d["innov"][4:10] = upd.innovation  # sun and B slots of the thesis layout
            nis_sun, nis_mag = self._nis()
            d["nis"][1], d["nis"][2] = nis_sun, nis_mag
            gain_rows = np.linalg.norm(upd.gain, axis=1)
            d["bias_gain"][0:3] = gain_rows[MAG]
            d["bias_gain"][3:6] = gain_rows[GYRO]
        return d
