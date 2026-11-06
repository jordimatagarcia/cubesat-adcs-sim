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
  cycle, disturbances) don't enter the estimate.
"""

import numpy as np
from numpy.typing import NDArray

from adcssim.adcs_ref.control import ModeController
from adcssim.adcs_ref.flae import flae
from adcssim.adcs_ref.limits import field_plausible, gyro_plausible, unit_plausible
from adcssim.adcs_ref.mekf import Mekf
from adcssim.adcs_ref.usque import ATT, GYRO, MAG, Usque
from adcssim.backends.base import AdcsContext, AdcsInput, AdcsMode, AdcsOutput, AdcsStatus
from adcssim.backends.config import AdcsConfig
from adcssim.frames import quat_rotate

FloatArray = NDArray[np.float64]
UT = 1e6

FAULT_NONE, FAULT_GYRO, FAULT_MAG = 0, 1, 2


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
        self.ctx = ctx
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
        mag_ut = np.asarray(inp.mag_b, dtype=np.float64) * UT if mag_ok else np.zeros(3)
        mag_ref_ut = np.asarray(inp.mag_ref_i, dtype=np.float64) * UT if mag_ref_ok else np.zeros(3)
        sun_b = np.asarray(inp.sun_b, dtype=np.float64) if sun_ok else np.zeros(3)
        sun_ref = np.asarray(inp.sun_ref_i, dtype=np.float64) if sun_ref_ok else np.zeros(3)
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
            return self._finish(inp, status, gyro, mag_ut, mag_ok, res.q)

        # Propagate with the mean of the previous and current rate readings: the body rate
        # changes within a step (Euler coupling while tumbling), and holding the previous
        # reading constant leaves an O(ω̇ dt²) attitude error per step.
        rate = 0.5 * (self.gyro_prev + gyro)
        ok = self.filter.step(rate, self.ctx.dt, sun_b, sun_ref, sun_ok, mag_ut, mag_ref_ut, mag_ok)
        if not ok:
            status |= AdcsStatus.COVARIANCE_RESET
        self._detect_faults(sun_ok)
        if self.fault != FAULT_NONE:
            status |= AdcsStatus.BIAS_FAULT
        self.gyro_prev = gyro
        return self._finish(inp, status, gyro, mag_ut, mag_ok, None)

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
        )
        cfg = self.config
        p = f.p
        p_diag = np.full(19, np.nan)
        p_diag[0:3] = np.diag(p)[ATT]  # attitude error variance per axis (rad²)
        p_diag[10:13] = cfg.sigma_gyro_noise**2 + np.diag(p)[GYRO]  # rate = gyro − bias
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
