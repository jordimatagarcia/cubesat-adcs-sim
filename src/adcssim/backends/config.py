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
    """Attitude uncertainty (rad) added on every detected bias fault."""
    sigma_outage_accel: float = 1e-3
    """Angular-acceleration uncertainty (rad/s²) while the gyro is unavailable: the rate is
    then propagated with Euler's torque-free equations, and this covers what they leave
    out (coil and disturbance torques, inertia errors). 1e-3 rad/s² is about the largest
    magnetorquer torque on a 3U CubeSat."""
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
        if not self.sigma_outage_accel >= 0.0:
            raise ValueError("sigma_outage_accel must be non-negative")
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

    def validate(self, dt: float, inertia: Any, max_dipole: Any) -> None:
        """The configuration rules shared by every implementation (Python, C, Rust): what
        ``adcs_init`` accepts. Raises ``ValueError`` naming the first rule broken."""
        problems = config_problems(self, dt, inertia, max_dipole)
        if problems:
            raise ValueError("invalid ADCS configuration: " + "; ".join(problems))

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


MAX_PROCESS_STEPS = 1000
"""Largest dt / process_dt: bounds the RK4 steps per filter step (and so its run time)."""
MAX_FLAE_ITERATIONS = 1000
ALPHA_RANGE = (1e-2, 1e2)
"""ukf_alpha: below ~1e-4 the weights lose precision (n + λ = 3α² computed as n + (3α² − n))."""
MAX_BDOT_GAIN = 1e9
MAX_PD_GAIN = 1e3
MAX_DT = 10.0


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def config_problems(cfg: AdcsConfig, dt: float, inertia: Any, max_dipole: Any) -> list[str]:
    """Every rule of ``AdcsConfig.validate`` that ``cfg`` breaks. The C (``config_valid`` in
    adcs-c/src/adcs.c) and Rust (``config_valid`` in adcs-rs/adcs-core/src/adcs.rs)
    libraries implement the same rules; tests/test_native.py checks they agree."""
    out: list[str] = []
    inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
    dipole = np.asarray(max_dipole, dtype=np.float64)
    if not (_finite(dt) and 0.0 < dt <= MAX_DT):
        out.append(f"dt must be in (0, {MAX_DT}] s")
    if not np.all(np.isfinite(inertia)):
        out.append("inertia must be finite")
    else:
        scale = float(np.max(np.abs(inertia)))
        if float(np.max(np.abs(inertia - inertia.T))) > 1e-9 * scale:
            out.append("inertia must be symmetric")
        elif not _positive_definite(inertia):
            out.append("inertia must be positive definite")
    if dipole.shape != (3,) or not (np.all(np.isfinite(dipole)) and np.all(dipole > 0.0)):
        out.append("max_dipole must be three positive values")
    w = (cfg.flae_weight_sun, cfg.flae_weight_mag)
    if not (_finite(*w) and min(w) >= 0.0 and _finite(w[0] + w[1]) and w[0] + w[1] > 0.0):
        out.append("FLAE weights must be finite, non-negative, with a positive sum")
    if not (_finite(cfg.flae_tolerance) and cfg.flae_tolerance > 0.0):
        out.append("flae_tolerance must be positive")
    if not 1 <= cfg.flae_max_iterations <= MAX_FLAE_ITERATIONS:
        out.append(f"flae_max_iterations must be in [1, {MAX_FLAE_ITERATIONS}]")
    if not (_finite(cfg.bdot_gain) and 0.0 <= cfg.bdot_gain <= MAX_BDOT_GAIN):
        out.append(f"bdot_gain must be in [0, {MAX_BDOT_GAIN:g}]")
    if not all(_finite(g) and 0.0 <= g <= MAX_PD_GAIN for g in (cfg.kp, cfg.kd)):
        out.append(f"kp and kd must be in [0, {MAX_PD_GAIN:g}]")
    rates = (cfg.detumble_exit_rate, cfg.detumble_enter_rate)
    if not (_finite(*rates) and 0.0 <= rates[0] <= rates[1]):
        out.append("need 0 <= detumble_exit_rate <= detumble_enter_rate")
    if not (_finite(cfg.settle_time) and cfg.settle_time >= 0.0):
        out.append("settle_time must be non-negative")
    q = np.asarray(cfg.q_target, dtype=np.float64)
    if not (np.all(np.isfinite(q)) and abs(float(np.linalg.norm(q)) - 1.0) <= 1e-6):
        out.append("q_target must be a unit quaternion")
    if cfg.variant == "thesis":
        if not (_finite(cfg.process_dt) and cfg.process_dt > 0.0):
            out.append("process_dt must be positive")
        elif not dt / cfg.process_dt <= MAX_PROCESS_STEPS:
            out.append(f"dt / process_dt must be at most {MAX_PROCESS_STEPS}")
        if not (_finite(cfg.ukf_alpha) and ALPHA_RANGE[0] <= cfg.ukf_alpha <= ALPHA_RANGE[1]):
            out.append(f"ukf_alpha must be in [{ALPHA_RANGE[0]:g}, {ALPHA_RANGE[1]:g}]")
        for name in ("p_x0", "p_b0", "v"):
            a = np.asarray(getattr(cfg, name))
            if not (np.all(np.isfinite(a)) and np.all(a > 0.0)):
                out.append(f"{name} must be positive")
        for name in ("w_x", "w_b"):
            a = np.asarray(getattr(cfg, name))
            if not (np.all(np.isfinite(a)) and np.all(a >= 0.0)):
                out.append(f"{name} must be non-negative")
    else:
        positive = (
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
            v = getattr(cfg, name)
            if not (_finite(v) and v > 0.0):
                out.append(f"{name} must be positive")
        for name in ("sigma_gyro_bias_walk", "sigma_mag_bias_walk", "sigma_outage_accel"):
            v = getattr(cfg, name)
            if not (_finite(v) and v >= 0.0):
                out.append(f"{name} must be non-negative")
        if cfg.fault_steps < 1:
            out.append("fault_steps must be at least 1")
    return out


def _positive_definite(m: FloatArray) -> bool:
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return False
    return True
