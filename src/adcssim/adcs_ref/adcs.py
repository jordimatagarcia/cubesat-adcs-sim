"""The reference ADCS: FLAE + two-stage UKF + controllers + mode manager.

Each step (one control period):

1. Check inputs; a reading marked valid that contains NaN/Inf is treated as missing.
2. FLAE on the raw sun and magnetometer readings gives an attitude measurement q (thesis
   §5.4.1), sign-aligned with the current estimate. With fewer than two vectors (eclipse)
   there is no attitude measurement this step.
3. TSUKF update with y = [q, sun, B, ω] (thesis §5.4.2). The process model integrates the
   state over the control period in RK4 steps of ``process_dt`` (thesis: 0.1 s).
4. Mode manager and control law; the dipole is saturated preserving its direction.

Differences from the thesis, all documented in docs/adcs.md:

- The filter is initialised from the first measurements (q from FLAE) instead of the
  identity attitude. The thesis appendix has this as a commented-out alternative.
- In eclipse FLAE uses the filter's own sun vector, propagated to the current step, with
  the real magnetometer reading ("using the filtered data from the Kalman filter", thesis
  abstract). Before the filter has started there is no such vector, so the ADCS waits for
  the sun instead of running FLAE on a zeroed sun vector (thesis case 3a).
- The process model's rotation matrix uses the FLAE attitude when available (thesis) and the
  filter's own attitude otherwise.
"""

import numpy as np
from numpy.typing import NDArray

from adcssim.adcs_ref.control import bdot, quaternion_feedback, saturate, torque_to_dipole
from adcssim.adcs_ref.flae import flae
from adcssim.adcs_ref.tsukf import TwoStageUkf
from adcssim.backends.base import AdcsContext, AdcsInput, AdcsMode, AdcsOutput, AdcsStatus
from adcssim.backends.config import NB, NX, AdcsConfig
from adcssim.frames import quat_to_dcm

FloatArray = NDArray[np.float64]

UT = 1e6
"""Tesla → microtesla: the filter's unit for the magnetic field."""

# State layout
Q, SUN, MAG, W = slice(0, 4), slice(4, 7), slice(7, 10), slice(10, 13)


def fault_matrices() -> tuple[FloatArray, FloatArray]:
    """F_a = 0 and F_s mapping the magnetometer bias onto B and the gyro bias onto ω."""
    fa = np.zeros((NX, NB))
    fs = np.zeros((NX, NB))
    fs[MAG, 0:3] = np.eye(3)
    fs[W, 3:6] = np.eye(3)
    return fa, fs


def _normalize_q(x: FloatArray) -> FloatArray:
    out = x.copy()
    out[Q] /= np.linalg.norm(out[Q])
    return out


def _normalize_q_sun(x: FloatArray) -> FloatArray:
    out = _normalize_q(x)
    n = np.linalg.norm(out[SUN])
    if n > 0.0:
        out[SUN] /= n
    return out


def _cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """Row-wise cross product of (K, 3) batches with a (3,) or (K, 3) operand."""
    return np.cross(a, b)


def process_model(
    xs: FloatArray,
    dt: float,
    process_dt: float,
    rot: FloatArray,
    sun_i: FloatArray,
    mag_i_ut: FloatArray,
    torque: FloatArray,
    inertia: FloatArray,
    inertia_inv: FloatArray,
) -> FloatArray:
    """Eq. 126 integrated over ``dt`` with RK4 steps of at most ``process_dt``, for a batch
    of states (K, 13).

    One deliberate change from eq. 126: the body sun and field vectors rotate as
    ``−ω × v`` using the state's own vectors, where the thesis used ``−ω × A(q_FLAE) v_i``.
    With the thesis form, an attitude error in eclipse makes the predicted field wrong, the
    filter attributes the magnetometer innovation to ω and the gyro bias, and the estimate
    diverges (docs/adcs.md). ``rot``, ``sun_i`` and ``mag_i_ut`` are kept in the signature
    for that comparison and are unused here."""
    del rot, sun_i, mag_i_ut

    def xdot(x: FloatArray) -> FloatArray:
        q = x[:, Q] / np.linalg.norm(x[:, Q], axis=1, keepdims=True)
        w = x[:, W]
        out = np.empty_like(x)
        out[:, 0:3] = 0.5 * (q[:, 3:4] * w + _cross(q[:, 0:3], w))
        out[:, 3] = -0.5 * np.sum(q[:, 0:3] * w, axis=1)
        out[:, SUN] = -_cross(w, x[:, SUN])
        out[:, MAG] = -_cross(w, x[:, MAG])
        out[:, W] = (_cross(-w, w @ inertia.T) + torque) @ inertia_inv.T
        return out

    n = max(1, round(dt / process_dt))
    h = dt / n
    x = xs.copy()
    for _ in range(n):
        k1 = h * xdot(x)
        k2 = h * xdot(x + 0.5 * k1)
        k3 = h * xdot(x + 0.5 * k2)
        k4 = h * xdot(x + k3)
        x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x


def propagate_body_vector(v: FloatArray, w: FloatArray, dt: float) -> FloatArray:
    """A fixed inertial direction seen from a body rotating at ``w`` for ``dt``:
    ``v(t+dt) = R(−w dt) v(t)`` (Rodrigues' formula), renormalised."""
    angle = float(np.linalg.norm(w)) * dt
    if angle < 1e-12:
        out = v.copy()
    else:
        k = w / np.linalg.norm(w)
        c, s = np.cos(angle), np.sin(angle)
        out = v * c - np.cross(k, v) * s + k * (k @ v) * (1.0 - c)
    n = float(np.linalg.norm(out))
    return out / n if n > 0.0 else out


def _finite(v: FloatArray) -> bool:
    return bool(np.all(np.isfinite(v)))


class ReferenceAdcs:
    """The ``python-ref`` backend."""

    name = "python-ref"

    def __init__(self) -> None:
        self.ctx: AdcsContext | None = None
        self.filter: TwoStageUkf | None = None
        self.mode = AdcsMode.OFF
        self.mode_time = 0.0
        self.m_prev = np.zeros(3)
        self.b_prev: FloatArray | None = None
        self.last_flae_q: FloatArray | None = None

    @property
    def config(self) -> AdcsConfig:
        assert self.ctx is not None
        return self.ctx.config

    def reset(self, ctx: AdcsContext) -> None:
        cfg = ctx.config
        fa, fs = fault_matrices()
        self.ctx = ctx
        self.inertia = np.asarray(ctx.inertia, dtype=np.float64)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.filter = TwoStageUkf(fa, fs, cfg.w_x, cfg.w_b, cfg.v, cfg.ukf_alpha)
        self.mode = AdcsMode.ESTIMATE
        self.mode_time = 0.0
        self.m_prev = np.zeros(3)
        self.b_prev = None
        self.last_flae_q = None
        self.initialised = False

    # ------------------------------------------------------------------------------------
    def step(self, inp: AdcsInput) -> AdcsOutput:
        if self.ctx is None or self.filter is None:
            return AdcsOutput(status=int(AdcsStatus.NOT_INITIALISED))
        cfg = self.config
        status = AdcsStatus.OK

        # Sensor readings and on-board reference models are checked separately: in eclipse
        # the sun sensor has no reading but the ephemeris still gives the Sun direction.
        mag_ref_ok = _finite(inp.mag_ref_i) and float(np.linalg.norm(inp.mag_ref_i)) > 0.0
        sun_ref_ok = _finite(inp.sun_ref_i) and float(np.linalg.norm(inp.sun_ref_i)) > 0.0
        mag_ok = inp.mag_valid and _finite(inp.mag_b)
        sun_ok = inp.sun_valid and _finite(inp.sun_b)
        gyro_ok = inp.gyro_valid and _finite(inp.gyro_b)
        if (
            (inp.mag_valid and not mag_ok)
            or (inp.sun_valid and not sun_ok)
            or (inp.gyro_valid and not gyro_ok)
        ):
            status |= AdcsStatus.BAD_INPUT
        mag_ut = np.asarray(inp.mag_b) * UT if mag_ok else np.zeros(3)
        mag_ref_ut = np.asarray(inp.mag_ref_i) * UT if mag_ref_ok else np.zeros(3)
        sun_b = np.asarray(inp.sun_b) if sun_ok else np.zeros(3)
        gyro = np.asarray(inp.gyro_b) if gyro_ok else np.zeros(3)

        # Attitude measurement from FLAE (needs two vectors). Without a sun reading, the
        # filter's sun vector, propagated over this step, stands in for it.
        q_meas: FloatArray | None = None
        sun_for_flae: FloatArray | None = sun_b if sun_ok else None
        if sun_for_flae is None and self.initialised and self.filter.state is not None:
            sun_for_flae = propagate_body_vector(
                self.filter.state.x[SUN], self.filter.state.x[W], self.ctx.dt
            )
        if mag_ok and mag_ref_ok and sun_ref_ok and sun_for_flae is not None:
            prev = self.filter.state.x[Q] if self.filter.state is not None else self.last_flae_q
            res = flae(
                np.vstack([sun_for_flae, mag_ut / np.linalg.norm(mag_ut)]),
                np.vstack([inp.sun_ref_i, mag_ref_ut / np.linalg.norm(mag_ref_ut)]),
                [cfg.flae_weight_sun, cfg.flae_weight_mag],
                q_prev=prev,
                tolerance=cfg.flae_tolerance,
                max_iterations=cfg.flae_max_iterations,
            )
            if res.ok:
                q_meas = res.q
                self.last_flae_q = res.q
            else:
                status |= AdcsStatus.FLAE_FAILED
        else:
            status |= AdcsStatus.NO_ATTITUDE_FIX

        if not self.initialised:
            if not sun_ok:
                q_meas = None  # never start from an attitude that has no real sun reading
            if q_meas is None or not gyro_ok:
                return self._output(status, mode=AdcsMode.OFF)
            x0 = np.concatenate([q_meas, sun_b, mag_ut, gyro])
            self.filter.reset(x0, cfg.p_x0, np.zeros(NB), cfg.p_b0)
            self.initialised = True
            self.b_prev = None
            return self._finish(inp, status)

        state = self.filter.state
        assert state is not None
        rot = quat_to_dcm(q_meas if q_meas is not None else state.x[Q])
        torque = np.cross(self.m_prev, mag_ut / UT) if mag_ok else np.zeros(3)
        dt = self.ctx.dt

        def f(xs: FloatArray) -> FloatArray:
            return process_model(
                xs,
                dt,
                cfg.process_dt,
                rot,
                np.asarray(inp.sun_ref_i, dtype=np.float64) if sun_ref_ok else rot.T @ state.x[SUN],
                mag_ref_ut if mag_ref_ok else rot.T @ state.x[MAG],
                torque,
                self.inertia,
                self.inertia_inv,
            )

        y = np.concatenate([q_meas if q_meas is not None else np.zeros(4), sun_b, mag_ut, gyro])
        available = np.repeat([q_meas is not None, sun_ok, mag_ok, gyro_ok], [4, 3, 3, 3])
        ok = self.filter.step(
            f, y, available, normalize_pred=_normalize_q, normalize_out=_normalize_q_sun
        )
        if not ok:
            status |= AdcsStatus.COVARIANCE_RESET
        return self._finish(inp, status)

    # ------------------------------------------------------------------------------------
    def _finish(self, inp: AdcsInput, status: AdcsStatus) -> AdcsOutput:
        """Mode manager and control law, after the estimate is updated."""
        assert self.ctx is not None and self.filter is not None and self.filter.state
        cfg = self.config
        x = self.filter.state.x
        w = x[W]
        b_body = x[MAG] / UT
        rate = float(np.linalg.norm(w))
        dt = self.ctx.dt

        if status & (AdcsStatus.BAD_INPUT | AdcsStatus.COVARIANCE_RESET):
            mode = AdcsMode.SAFE
        elif not cfg.control_enabled:
            mode = AdcsMode.ESTIMATE
        elif self.mode in (AdcsMode.ESTIMATE, AdcsMode.POINT, AdcsMode.SAFE, AdcsMode.OFF):
            if rate > cfg.detumble_enter_rate:
                mode = AdcsMode.DETUMBLE
            elif self.mode == AdcsMode.POINT or (
                self.mode == AdcsMode.ESTIMATE and self.mode_time >= cfg.settle_time
            ):
                mode = AdcsMode.POINT
            else:
                mode = AdcsMode.ESTIMATE
        else:  # DETUMBLE
            mode = AdcsMode.ESTIMATE if rate < cfg.detumble_exit_rate else AdcsMode.DETUMBLE
        self.mode_time = self.mode_time + dt if mode == self.mode else 0.0
        self.mode = mode

        m = np.zeros(3)
        if mode == AdcsMode.DETUMBLE and self.b_prev is not None:
            m = bdot((b_body - self.b_prev) / dt, cfg.bdot_gain)
        elif mode == AdcsMode.POINT:
            tau = quaternion_feedback(x[Q], cfg.q_target, w, cfg.kp, cfg.kd)
            m = torque_to_dipole(tau, b_body)
        m, saturated = saturate(m, np.asarray(self.ctx.max_dipole, dtype=np.float64))
        if saturated:
            status |= AdcsStatus.DIPOLE_SATURATED
        self.b_prev = b_body
        self.m_prev = m
        return self._output(status, mode, m)

    def _output(
        self, status: AdcsStatus, mode: AdcsMode, m: FloatArray | None = None
    ) -> AdcsOutput:
        s = self.filter.state if self.filter is not None and self.initialised else None
        if s is None:
            return AdcsOutput(mode=mode, status=int(status))
        return AdcsOutput(
            q_est=s.x[Q].copy(),
            w_est=s.x[W].copy(),
            bias_mag=s.b[0:3] / UT,
            bias_gyro=s.b[3:6].copy(),
            p_diag=np.concatenate([np.diag(s.px), np.diag(s.pb)]),
            m_cmd=np.zeros(3) if m is None else m,
            mode=mode,
            status=int(status),
        )
