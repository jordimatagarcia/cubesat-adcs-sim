"""Error-state unscented attitude filter (USQUE), J. L. Crassidis and F. L. Markley,
"Unscented Filtering for Spacecraft Attitude Estimation", JGCD 26(4), 2003. Extended with
a magnetometer bias state.

State (9): ``x = [δp, β_g, β_m]``: the attitude error as generalised Rodrigues parameters
(GRP, a = 1, f = 2(a + 1) = 4, chosen so that ``δp`` equals the rotation angle vector for
small errors, in radians) around a reference
quaternion ``q_ref``, the gyro bias (rad/s) and the magnetometer bias (µT). The gyro is the
rate input, so no rigid-body dynamics are modelled.

Measurements: ``y_sun = A(q) s_i`` (unit vector) and ``y_B = A(q) B_i + β_m`` (µT). Either
can be unavailable; like the thesis filter, an unavailable measurement gets an effectively
infinite variance and its predicted value, so the filter stays fixed-size.

Each step propagates the sigma points from the previous step to this one with the measured
rate (the caller passes the mean of the two readings), updates with this step's vectors,
then folds the attitude error into ``q_ref`` and resets it to zero.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from adcssim.adcs_ref.tsukf import UNAVAILABLE_VARIANCE, robust_cholesky
from adcssim.frames import quat_conj, quat_mul, quat_to_dcm

FloatArray = NDArray[np.float64]

GRP_A = 1.0
GRP_F = 2.0 * (GRP_A + 1.0)
NX = 9
ATT, GYRO, MAG = slice(0, 3), slice(3, 6), slice(6, 9)


def grp_to_quat(p: FloatArray) -> FloatArray:
    """Error quaternion(s) from GRPs, Crassidis & Markley 2003 eq. 32."""
    p = np.asarray(p, dtype=np.float64)
    n2 = np.sum(p * p, axis=-1, keepdims=True)
    q4 = (-GRP_A * n2 + GRP_F * np.sqrt(GRP_F**2 + (1.0 - GRP_A**2) * n2)) / (GRP_F**2 + n2)
    qv = (GRP_A + q4) * p / GRP_F
    return np.concatenate([qv, q4], axis=-1)


def quat_to_grp(q: FloatArray) -> FloatArray:
    """GRPs of error quaternion(s), eq. 31; the sign is chosen so that q4 >= 0."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., 3:] < 0.0, -q, q)
    return np.asarray(GRP_F * q[..., :3] / (GRP_A + q[..., 3:]), dtype=np.float64)


def rotation_quat(w: FloatArray, dt: float) -> FloatArray:
    """Quaternion(s) of a body rotating at constant rate ``w`` for ``dt`` (exact)."""
    w = np.asarray(w, dtype=np.float64)
    rate = np.linalg.norm(w, axis=-1, keepdims=True)
    half = 0.5 * rate * dt
    # sin(x)/x expanded near 0 to avoid dividing by zero.
    sinc = np.where(rate > 1e-12, np.sin(half) / np.where(rate > 1e-12, rate, 1.0), 0.5 * dt)
    return np.concatenate([w * sinc, np.cos(half)], axis=-1)


@dataclass(frozen=True)
class UsqueUpdate:
    innovation: FloatArray
    """y − ŷ for [sun, B], NaN where unavailable."""
    innovation_cov: FloatArray
    gain: FloatArray
    available: NDArray[np.bool_]


class Usque:
    def __init__(
        self,
        lam: float,
        sigma_gyro_noise: float,
        sigma_gyro_bias_walk: float,
        sigma_mag_noise: float,
        sigma_mag_bias_walk: float,
        sigma_sun_noise: float,
    ) -> None:
        self.lam = lam
        self.wm0 = lam / (NX + lam)
        self.wi = 1.0 / (2.0 * (NX + lam))
        self.sigma_gyro_noise = sigma_gyro_noise
        self.sigma_gyro_bias_walk = sigma_gyro_bias_walk
        self.sigma_mag_bias_walk = sigma_mag_bias_walk
        self.r = np.diag(np.r_[np.full(3, sigma_sun_noise**2), np.full(3, sigma_mag_noise**2)])
        self.q_ref = np.array([0.0, 0.0, 0.0, 1.0])
        self.x = np.zeros(NX)
        self.p = np.eye(NX)
        self.p0 = np.eye(NX)
        self.last_update: UsqueUpdate | None = None

    def reset(self, q0: FloatArray, p0_diag: FloatArray) -> None:
        self.q_ref = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
        self.x = np.zeros(NX)
        self.p0 = np.diag(p0_diag)
        self.p = self.p0.copy()

    def process_noise(self, dt: float) -> FloatArray:
        """Q̄ of eq. 42 (added before drawing sigma points and again after propagating),
        with the magnetometer-bias random walk appended."""
        sv2 = self.sigma_gyro_noise**2 * dt  # angle random walk density (rad²/s)
        su2 = self.sigma_gyro_bias_walk**2
        sm2 = self.sigma_mag_bias_walk**2
        diag = np.r_[
            np.full(3, sv2 - su2 * dt**2 / 6.0),
            np.full(3, su2),
            np.full(3, sm2),
        ]
        return np.asarray(0.5 * dt * np.diag(diag), dtype=np.float64)

    def _mean(self, s: FloatArray) -> FloatArray:
        return np.asarray(self.wm0 * s[0] + self.wi * s[1:].sum(axis=0), dtype=np.float64)

    def _cov(self, a: FloatArray, b: FloatArray) -> FloatArray:
        return np.asarray(
            self.wm0 * np.outer(a[0], b[0]) + self.wi * a[1:].T @ b[1:], dtype=np.float64
        )

    def step(
        self,
        gyro: FloatArray,
        dt: float,
        sun_b: FloatArray,
        sun_i: FloatArray,
        sun_available: bool,
        mag_ut: FloatArray,
        mag_i_ut: FloatArray,
        mag_available: bool,
        extra_attitude_var: float = 0.0,
        freeze_gyro_bias: bool = False,
    ) -> bool:
        """Propagate over ``dt`` with the measured rate ``gyro`` (held constant over the
        step), then update. ``extra_attitude_var`` and ``freeze_gyro_bias`` as in
        :meth:`Mekf.step`. Returns False if the covariance had to be reset."""
        ok = True
        q_bar = self.process_noise(dt)
        lower = robust_cholesky((NX + self.lam) * (self.p + q_bar))
        if lower is None:
            self.p, ok = self.p0.copy(), False
            lower = np.linalg.cholesky((NX + self.lam) * (self.p + q_bar))
        spread = lower.T
        chi = np.vstack([self.x, self.x + spread, self.x - spread])  # (2n + 1, 9)

        # Propagate each sigma point's attitude with its own bias-corrected rate.
        dq = grp_to_quat(chi[:, ATT])
        q_i = quat_mul(dq, self.q_ref)
        q_prop = quat_mul(rotation_quat(gyro - chi[:, GYRO], dt), q_i)
        q0_prop = q_prop[0]
        chi_prop = chi.copy()
        chi_prop[:, ATT] = quat_to_grp(quat_mul(q_prop, quat_conj(q0_prop)))
        x_pred = self._mean(chi_prop)
        dev = chi_prop - x_pred
        p_pred = self._cov(dev, dev) + q_bar
        p_pred[ATT, ATT] += np.eye(3) * extra_attitude_var

        # Predicted measurements.
        a = quat_to_dcm(q_prop)  # (2n + 1, 3, 3)
        gamma = np.concatenate([a @ sun_i, a @ mag_i_ut + chi_prop[:, MAG]], axis=1)  # (2n + 1, 6)
        y_hat = self._mean(gamma)
        available = np.repeat([sun_available, mag_available], 3)
        y = np.where(available, np.r_[sun_b, mag_ut], y_hat)
        r = self.r.copy()
        r[~available, ~available] = UNAVAILABLE_VARIANCE
        gdev = gamma - y_hat
        p_yy = self._cov(gdev, gdev) + r
        p_xy = self._cov(dev, gdev)
        gain = np.linalg.solve(p_yy.T, p_xy.T).T
        innov = y - y_hat
        if freeze_gyro_bias:
            gain[GYRO, :] = 0.0
            # general form, valid for a sub-optimal gain
            p_upd = p_pred - gain @ p_xy.T - p_xy @ gain.T + gain @ p_yy @ gain.T
        else:
            p_upd = p_pred - gain @ p_yy @ gain.T
        x_upd = x_pred + gain @ innov
        p_upd = 0.5 * (p_upd + p_upd.T)
        self.last_update = UsqueUpdate(
            innovation=np.where(available, innov, np.nan),
            innovation_cov=p_yy,
            gain=gain,
            available=available,
        )

        if not (np.all(np.isfinite(x_upd)) and np.all(np.isfinite(p_upd))):
            # Keep the propagated attitude and the bias estimates, like the MEKF.
            self.q_ref = q0_prop / np.linalg.norm(q0_prop)
            self.p = self.p0.copy()
            return False
        # Fold the attitude error into the reference quaternion.
        self.q_ref = quat_mul(grp_to_quat(x_upd[ATT]), q0_prop)
        self.q_ref /= np.linalg.norm(self.q_ref)
        x_upd[ATT] = 0.0
        self.x, self.p = x_upd, p_upd
        return ok

    def inflate(self, attitude: float = 0.0, gyro_bias: float = 0.0, mag_bias: float = 0.0) -> None:
        """Add uncertainty after a detected fault (σ values; attitude in rad)."""
        self.p[ATT, ATT] += np.eye(3) * attitude**2
        self.p[GYRO, GYRO] += np.eye(3) * gyro_bias**2
        self.p[MAG, MAG] += np.eye(3) * mag_bias**2
