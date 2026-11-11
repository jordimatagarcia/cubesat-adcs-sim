"""Multiplicative extended Kalman filter (MEKF), Markley & Crassidis (2014) §6.2, on the same
model as the USQUE filter (``usque.py``): attitude error δθ (3) around a reference quaternion,
gyro bias (3), magnetometer bias (µT, 3); the gyro is the rate input; sun and magnetometer
vectors are measured against the attitude.

The industry-standard baseline: linearised propagation and measurement (Jacobians) instead of
sigma points. Same interface as :class:`~adcssim.adcs_ref.usque.Usque`.
"""

import numpy as np
from numpy.typing import NDArray

from adcssim.adcs_ref.tsukf import UNAVAILABLE_VARIANCE
from adcssim.adcs_ref.usque import ATT, GYRO, MAG, NX, UsqueUpdate, rotation_quat
from adcssim.frames import quat_mul, quat_to_dcm, skew

FloatArray = NDArray[np.float64]


def _rotation_matrix(w: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
    """Exact Φ11 = exp(−[ω×]dt) and Φ12 = −∫₀^dt exp(−[ω×]s) ds for a constant rate."""
    wx = skew(w)
    rate = float(np.linalg.norm(w))
    if rate < 1e-9:
        return np.eye(3) - wx * dt, -np.eye(3) * dt + 0.5 * wx * dt**2
    th = rate * dt
    phi11 = np.eye(3) - np.sin(th) / rate * wx + (1.0 - np.cos(th)) / rate**2 * wx @ wx
    phi12 = (
        wx * (1.0 - np.cos(th)) / rate**2 - np.eye(3) * dt - wx @ wx * (th - np.sin(th)) / rate**3
    )
    return phi11, phi12


class Mekf:
    def __init__(
        self,
        sigma_gyro_noise: float,
        sigma_gyro_bias_walk: float,
        sigma_mag_noise: float,
        sigma_mag_bias_walk: float,
        sigma_sun_noise: float,
    ) -> None:
        self.sigma_gyro_noise = sigma_gyro_noise
        self.sigma_gyro_bias_walk = sigma_gyro_bias_walk
        self.sigma_mag_bias_walk = sigma_mag_bias_walk
        self.r = np.diag(np.r_[np.full(3, sigma_sun_noise**2), np.full(3, sigma_mag_noise**2)])
        self.q_ref = np.array([0.0, 0.0, 0.0, 1.0])
        self.x = np.zeros(NX)  # [0 (attitude error is always folded in), β_g, β_m]
        self.p = np.eye(NX)
        self.p0 = np.eye(NX)
        self.last_update: UsqueUpdate | None = None

    def reset(self, q0: FloatArray, p0_diag: FloatArray) -> None:
        self.q_ref = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
        self.x = np.zeros(NX)
        self.p0 = np.diag(p0_diag)
        self.p = self.p0.copy()

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
        """Propagate with the rate ``gyro`` and update. ``extra_attitude_var`` (rad²) is added
        to the attitude variance (gyro outage); with ``freeze_gyro_bias`` the update leaves
        the gyro-bias estimate unchanged (its gain rows are zero; the Joseph form keeps the
        covariance correct for that sub-optimal gain). Returns False if the result was not
        finite and the covariance was reset."""
        # ---- propagation ------------------------------------------------------------
        w = gyro - self.x[GYRO]
        self.q_ref = quat_mul(rotation_quat(w, dt), self.q_ref)
        phi11, phi12 = _rotation_matrix(w, dt)
        phi = np.eye(NX)
        phi[ATT, ATT] = phi11
        phi[ATT, GYRO] = phi12
        sv2 = self.sigma_gyro_noise**2 * dt  # angle random walk density (rad²/s)
        su2 = self.sigma_gyro_bias_walk**2
        q = np.zeros((NX, NX))
        q[ATT, ATT] = np.eye(3) * (sv2 * dt + su2 * dt**3 / 3.0)
        q[ATT, GYRO] = q[GYRO, ATT] = -np.eye(3) * su2 * dt**2 / 2.0
        q[GYRO, GYRO] = np.eye(3) * su2 * dt
        q[MAG, MAG] = np.eye(3) * self.sigma_mag_bias_walk**2 * dt
        p_pred = phi @ self.p @ phi.T + q
        for i in range(3):
            p_pred[i, i] += extra_attitude_var

        # ---- update -------------------------------------------------------------------
        a = quat_to_dcm(self.q_ref)
        sun_hat = a @ sun_i
        mag_body = a @ mag_i_ut
        y_hat = np.r_[sun_hat, mag_body + self.x[MAG]]
        h = np.zeros((6, NX))
        h[0:3, ATT] = skew(sun_hat)
        h[3:6, ATT] = skew(mag_body)
        h[3:6, MAG] = np.eye(3)
        available = np.repeat([sun_available, mag_available], 3)
        y = np.where(available, np.r_[sun_b, mag_ut], y_hat)
        r = self.r.copy()
        r[~available, ~available] = UNAVAILABLE_VARIANCE
        s = h @ p_pred @ h.T + r
        gain = np.linalg.solve(s.T, (p_pred @ h.T).T).T
        if freeze_gyro_bias:
            gain[GYRO, :] = 0.0
        innov = y - y_hat
        dx = gain @ innov
        ikh = np.eye(NX) - gain @ h
        p_upd = ikh @ p_pred @ ikh.T + gain @ r @ gain.T  # Joseph form
        p_upd = 0.5 * (p_upd + p_upd.T)
        self.last_update = UsqueUpdate(
            innovation=np.where(available, innov, np.nan),
            innovation_cov=s,
            gain=gain,
            available=available,
        )
        if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(p_upd))):
            self.p = self.p0.copy()
            return False
        dq = np.r_[0.5 * dx[ATT], 1.0]
        self.q_ref = quat_mul(dq / np.linalg.norm(dq), self.q_ref)
        self.q_ref /= np.linalg.norm(self.q_ref)
        self.x[GYRO] += dx[GYRO]
        self.x[MAG] += dx[MAG]
        self.p = p_upd
        return True

    def inflate(self, attitude: float = 0.0, gyro_bias: float = 0.0, mag_bias: float = 0.0) -> None:
        self.p[ATT, ATT] += np.eye(3) * attitude**2
        self.p[GYRO, GYRO] += np.eye(3) * gyro_bias**2
        self.p[MAG, MAG] += np.eye(3) * mag_bias**2
