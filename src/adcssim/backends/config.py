"""ADCS configuration, mirroring ``adcs_config`` in ``adcs-c/include/adcs.h``.

Defaults are the thesis tuning (§5.4.2). The filter handles the magnetic field in microtesla,
the unit that tuning was made in: ``V`` for the field is 1 µT², i.e. the 1 µT magnetometer
noise of the thesis cases. All other values are SI.
"""

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

NX = 13
"""State size: q(4), sun(3), B(3), ω(3)."""
NB = 6
"""Bias size: magnetometer(3), gyroscope(3)."""

STATE_GROUPS = (4, 3, 3, 3)
BIAS_GROUPS = (3, 3)


def expand(groups: tuple[int, ...], values: Any, name: str) -> FloatArray:
    """A diagonal given per group (e.g. ``[q, sun, B, ω]``) or element by element."""
    arr = np.asarray(values, dtype=np.float64)
    n = sum(groups)
    if arr.shape == (len(groups),):
        arr = np.repeat(arr, groups)
    if arr.shape != (n,):
        raise ValueError(f"{name} needs {len(groups)} group values or {n} values")
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and non-negative")
    return arr


def _groups(groups: tuple[int, ...], *values: float) -> FloatArray:
    return np.repeat(np.array(values), groups)


VARIANTS = ("thesis", "improved", "mekf")
"""``thesis``: the algorithm as reproduced from the thesis (docs/adcs.md), frozen together
with its golden vectors and results report. ``improved``: the error-state UKF designed from
the diagnosis (docs/results/diagnosis.md); its settings are the ``usque_*``, ``sigma_*``,
``p0_*`` and ``fault_*`` fields below. ``mekf``: the same model and settings with a
multiplicative extended Kalman filter instead of the UKF, as the industry-standard
baseline."""


@dataclass(frozen=True)
class AdcsConfig:
    variant: str = "thesis"
    # FLAE
    flae_weight_sun: float = 1.0
    flae_weight_mag: float = 1.0
    flae_tolerance: float = 1e-3
    flae_max_iterations: int = 20
    # Two-stage UKF (thesis §5.4.2)
    process_dt: float = 0.1
    ukf_alpha: float = 1.0
    p_x0: FloatArray = field(default_factory=lambda: _groups(STATE_GROUPS, 1e-4, 1e-2, 1e-7, 5e-7))
    p_b0: FloatArray = field(default_factory=lambda: _groups(BIAS_GROUPS, 1e-12, 1e-7))
    w_x: FloatArray = field(default_factory=lambda: _groups(STATE_GROUPS, 1e-8, 1e-4, 2e-2, 1e-9))
    w_b: FloatArray = field(default_factory=lambda: _groups(BIAS_GROUPS, 1e-11, 1e-6))
    v: FloatArray = field(default_factory=lambda: _groups(STATE_GROUPS, 1e-4, 1e-2, 1.0, 5e-6))
    # Improved variant: error-state UKF (USQUE, Crassidis & Markley 2003). Noise values are
    # standard deviations of the sensors as seen by the filter, per control period.
    usque_lambda: float = 1.0
    """Sigma-point scaling λ (n + λ = 10 for the 9-state filter; all weights positive)."""
    sigma_gyro_noise: float = 5e-4
    """Gyro white noise per sample (rad/s)."""
    sigma_gyro_bias_walk: float = 3e-6
    """Gyro bias random walk (rad/s/√s)."""
    sigma_mag_noise: float = 0.3
    """Magnetometer noise (µT)."""
    sigma_mag_bias_walk: float = 1e-3
    """Magnetometer bias random walk (µT/√s)."""
    sigma_sun_noise: float = 5e-3
    """Sun sensor noise per vector component (rad)."""
    p0_attitude: float = 0.05
    """Initial attitude uncertainty after the FLAE initialisation (rad, per axis)."""
    p0_gyro_bias: float = 5e-3
    """Initial gyro bias uncertainty (rad/s)."""
    p0_mag_bias: float = 2.0
    """Initial magnetometer bias uncertainty (µT)."""
    fault_nis_threshold: float = 25.0
    """Per-step NIS of the sun or magnetometer measurement above which a bias fault is
    suspected (χ²(3) at 99.99% is 21.1)."""
    fault_steps: int = 3
    """Consecutive suspect steps before the bias covariance is inflated."""
    fault_gyro_bias_sigma: float = 0.1
    """Gyro bias uncertainty (rad/s) after a detected fault."""
    fault_mag_bias_sigma: float = 50.0
    """Magnetometer bias uncertainty (µT) after a detected fault."""
    fault_attitude_sigma: float = 0.05
    """Attitude uncertainty (rad) after a detected gyro fault."""
    # Control
    control_enabled: bool = False
    bdot_gain: float = 4e4
    kp: float = 2e-5
    kd: float = 2e-3
    q_target: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    detumble_enter_rate: float = math.radians(4.0)
    detumble_exit_rate: float = math.radians(1.0)
    settle_time: float = 60.0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown ADCS variant {self.variant!r} ({', '.join(VARIANTS)})")
        for name, groups in (
            ("p_x0", STATE_GROUPS),
            ("w_x", STATE_GROUPS),
            ("v", STATE_GROUPS),
            ("p_b0", BIAS_GROUPS),
            ("w_b", BIAS_GROUPS),
        ):
            object.__setattr__(self, name, expand(groups, getattr(self, name), name))
        q = np.asarray(self.q_target, dtype=np.float64)
        if q.shape != (4,) or not math.isclose(float(np.linalg.norm(q)), 1.0, rel_tol=1e-6):
            raise ValueError("q_target must be a unit quaternion")
        object.__setattr__(self, "q_target", q)
        if self.process_dt <= 0.0 or self.ukf_alpha <= 0.0 or self.flae_tolerance <= 0.0:
            raise ValueError("process_dt, ukf_alpha and flae_tolerance must be positive")
        positive = (
            "usque_lambda",
            "sigma_gyro_noise",
            "sigma_mag_noise",
            "sigma_sun_noise",
            "p0_attitude",
            "p0_gyro_bias",
            "p0_mag_bias",
            "fault_nis_threshold",
            "fault_gyro_bias_sigma",
            "fault_mag_bias_sigma",
            "fault_attitude_sigma",
        )
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.sigma_gyro_bias_walk < 0.0 or self.sigma_mag_bias_walk < 0.0:
            raise ValueError("bias random walks must be non-negative")
        if self.fault_steps < 1:
            raise ValueError("fault_steps must be at least 1")
        if self.flae_max_iterations < 1:
            raise ValueError("flae_max_iterations must be at least 1")
        if self.flae_weight_sun < 0.0 or self.flae_weight_mag < 0.0:
            raise ValueError("FLAE weights must be non-negative")
        if not 0.0 <= self.detumble_exit_rate <= self.detumble_enter_rate:
            raise ValueError("need 0 <= detumble_exit_rate <= detumble_enter_rate")

    @classmethod
    def from_table(cls, table: dict[str, Any]) -> "AdcsConfig":
        """Build from a scenario's ``[adcs]`` table (``backend`` excluded). Rates in the file
        are in °/s, as the key names say."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in table.items():
            if key in ("detumble_enter_rate_deg_s", "detumble_exit_rate_deg_s"):
                kwargs[key.removesuffix("_deg_s")] = math.radians(float(value))
            elif key in known and key not in ("detumble_enter_rate", "detumble_exit_rate"):
                kwargs[key] = value
            else:
                raise ValueError(f"unknown key 'adcs.{key}'")
        return replace(cls(), **kwargs)
