"""Magnetorquer control laws (thesis §4.2.3)."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from adcssim.backends.base import AdcsMode, AdcsStatus
from adcssim.backends.config import AdcsConfig
from adcssim.frames import quat_conj, quat_mul

FloatArray = NDArray[np.float64]


def bdot(b_dot: FloatArray, gain: float) -> FloatArray:
    """B-dot detumbling, eq. 114: ``m = −k Ḃ`` (Ḃ in the body frame, T/s)."""
    return -gain * b_dot


def quaternion_feedback(
    q: FloatArray, q_target: FloatArray, w: FloatArray, kp: float, kd: float
) -> FloatArray:
    """Quaternion-feedback PD torque, eq. 119: ``τ = −kp sign(δq4) δq1:3 − kd ω``.

    ``δq = q ⊗ q_d⁻¹`` (eq. 116). The sign term makes the controller take the shorter of
    the two rotations to the target.
    """
    dq = quat_mul(q, quat_conj(q_target))
    sign = 1.0 if dq[3] >= 0.0 else -1.0
    return np.asarray(-kp * sign * dq[:3] - kd * w)


def torque_to_dipole(torque: FloatArray, b_body: FloatArray) -> FloatArray:
    """Eq. 121: ``m = (B × τ) / |B|²``. Only the torque component perpendicular to B can be
    produced by magnetorquers; this gives the dipole that produces exactly that component."""
    b2 = float(b_body @ b_body)
    if b2 == 0.0:
        return np.zeros(3)
    return np.asarray(np.cross(b_body, torque) / b2)


def saturate(m: FloatArray, max_dipole: FloatArray) -> tuple[FloatArray, bool]:
    """Scale the whole dipole vector so no coil exceeds its limit, keeping its direction."""
    ratio = float(np.max(np.abs(m) / max_dipole))
    if ratio <= 1.0:
        return m, False
    return m / ratio, True


@dataclass
class ModeController:
    """Mode manager and control law, shared by all estimator variants.

    ``DETUMBLE`` above ``detumble_enter_rate`` until the rate drops below
    ``detumble_exit_rate``, then ``ESTIMATE`` (coils off) for ``settle_time``, then
    ``POINT``. ``SAFE`` (coils off) on a bad input or a filter reset. With control disabled
    the mode stays ``ESTIMATE``.
    """

    mode: AdcsMode = AdcsMode.ESTIMATE
    mode_time: float = 0.0
    b_prev: FloatArray | None = None
    m_prev: FloatArray = field(default_factory=lambda: np.zeros(3))

    def update(
        self,
        cfg: AdcsConfig,
        dt: float,
        max_dipole: FloatArray,
        q: FloatArray,
        w: FloatArray,
        b_body: FloatArray,
        status: AdcsStatus,
    ) -> tuple[AdcsMode, FloatArray, AdcsStatus]:
        """New mode and dipole command from the current estimate (``b_body`` in tesla)."""
        rate = float(np.linalg.norm(w))
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
            tau = quaternion_feedback(q, cfg.q_target, w, cfg.kp, cfg.kd)
            m = torque_to_dipole(tau, b_body)
        m, saturated = saturate(m, max_dipole)
        if saturated:
            status |= AdcsStatus.DIPOLE_SATURATED
        self.b_prev = b_body
        self.m_prev = m
        return mode, m, status
