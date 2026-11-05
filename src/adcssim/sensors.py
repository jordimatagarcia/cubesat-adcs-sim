"""Sensor models: magnetometer, gyroscope and sun sensors (thesis §4.1.2, §5.2).

Each sensor draws its noise from its own random generator, so adding or removing one sensor
never changes the noise sequence of another. All readings are in the body frame (sensor
frames are aligned with the body; mounting errors are modelled as misalignment).

Default ranges and resolutions come from the datasheets of the Foresail-1 sensors named in
the thesis: LIS3MDL magnetometer and L3GD20H gyroscope (±16 gauss / ±2000 °/s settings),
Hamamatsu S5990-1 PSD sun sensors with a 90° field of view.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from adcssim.frames import skew, unit

FloatArray = NDArray[np.float64]

MAG_RANGE_T = 1.6e-3
MAG_LSB_T = 1.0 / 1.711e7
GYRO_RANGE_RAD_S = math.radians(2000.0)
GYRO_LSB_RAD_S = math.radians(0.07)
PSD_HALF_FOV_RAD = math.radians(45.0)


def _vec(values: object = (0.0, 0.0, 0.0)) -> FloatArray:
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {a.shape}")
    return a


@dataclass(frozen=True)
class BiasWindow:
    """An extra bias that is present only for ``start_s <= t <= end_s``."""

    start_s: float
    end_s: float
    bias: FloatArray

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("bias window ends before it starts")
        object.__setattr__(self, "bias", _vec(self.bias))

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


@dataclass(frozen=True)
class SensorErrors:
    """Error model shared by the vector sensors (thesis eqs. 91–93).

    ``measured = K_no K_mis K_sf · x + bias(t) + noise``, then saturation and quantisation.
    """

    noise_std: float = 0.0
    """Standard deviation of the white noise on each axis (sensor units)."""
    bias: FloatArray = field(default_factory=_vec)
    """Constant bias (sensor units)."""
    bias_windows: tuple[BiasWindow, ...] = ()
    scale_factor: FloatArray = field(default_factory=_vec)
    """Scale-factor errors per axis: ``K_sf = diag(1 + sf)``."""
    misalignment: FloatArray = field(default_factory=_vec)
    """Small-angle misalignment (rad) about x, y, z: ``K_mis = I − [θ×]``."""
    nonorthogonality: FloatArray = field(default_factory=_vec)
    """Non-orthogonality angles (rad) of the axis pairs (xy, xz, yz)."""
    range: float = math.inf
    """Saturation level per axis (sensor units)."""
    lsb: float = 0.0
    """Quantisation step (sensor units); 0 disables quantisation."""

    def __post_init__(self) -> None:
        for name in ("bias", "scale_factor", "misalignment", "nonorthogonality"):
            object.__setattr__(self, name, _vec(getattr(self, name)))
        if self.noise_std < 0.0 or self.lsb < 0.0 or self.range <= 0.0:
            raise ValueError("noise_std and lsb must be >= 0 and range > 0")

    @property
    def matrix(self) -> FloatArray:
        """Combined error matrix ``K_no K_mis K_sf``."""
        k_sf = np.diag(1.0 + self.scale_factor)
        k_mis = np.eye(3) - skew(self.misalignment)
        xy, xz, yz = self.nonorthogonality
        k_no = np.array([[1.0, 0.0, 0.0], [xy, 1.0, 0.0], [xz, yz, 1.0]])
        return np.asarray(k_no @ k_mis @ k_sf)

    def bias_at(self, t: float) -> FloatArray:
        total = self.bias.copy()
        for window in self.bias_windows:
            if window.active(t):
                total += window.bias
        return total

    def finish(self, x: FloatArray, rng: np.random.Generator) -> tuple[FloatArray, bool]:
        """Add noise, saturate and quantise. Returns the reading and whether it's in range."""
        y = x + rng.normal(0.0, self.noise_std, 3) if self.noise_std > 0.0 else x.copy()
        in_range = bool(np.all(np.abs(y) <= self.range))
        y = np.clip(y, -self.range, self.range)
        if self.lsb > 0.0:
            y = np.round(y / self.lsb) * self.lsb
        return y, in_range


@dataclass(frozen=True)
class MagnetometerModel:
    """Three-axis magnetometer, thesis eq. 93: ``B = K (B_true + S_si B_true + b_hi) + b + ε``."""

    errors: SensorErrors = field(
        default_factory=lambda: SensorErrors(range=MAG_RANGE_T, lsb=MAG_LSB_T)
    )
    hard_iron: FloatArray = field(default_factory=_vec)
    """Hard-iron offset (T)."""
    soft_iron: FloatArray = field(default_factory=lambda: np.zeros((3, 3)))
    """Soft-iron matrix ``S_si`` (dimensionless)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_iron", _vec(self.hard_iron))
        s = np.asarray(self.soft_iron, dtype=np.float64)
        if s.shape != (3, 3):
            raise ValueError("soft_iron must be 3x3")
        object.__setattr__(self, "soft_iron", s)

    def true_bias(self, t: float) -> FloatArray:
        """The additive offset an estimator should find: scaled hard iron plus bias."""
        return np.asarray(self.errors.matrix @ self.hard_iron + self.errors.bias_at(t))

    def measure(
        self, t: float, b_body: FloatArray, rng: np.random.Generator
    ) -> tuple[FloatArray, bool]:
        x = self.errors.matrix @ ((np.eye(3) + self.soft_iron) @ b_body + self.hard_iron)
        return self.errors.finish(x + self.errors.bias_at(t), rng)


@dataclass
class GyroscopeModel:
    """Three-axis MEMS gyroscope, thesis eq. 91: ``ω = K ω_true + b(t) + ε``.

    The bias may also drift as a random walk with ``bias_walk_std`` rad/s/√s; this makes the
    model stateful, so call :meth:`measure` with increasing times.
    """

    errors: SensorErrors = field(
        default_factory=lambda: SensorErrors(range=GYRO_RANGE_RAD_S, lsb=GYRO_LSB_RAD_S)
    )
    bias_walk_std: float = 0.0
    _walk: FloatArray = field(default_factory=_vec, init=False, repr=False)
    _last_t: float | None = field(default=None, init=False, repr=False)

    def true_bias(self, t: float) -> FloatArray:
        return np.asarray(self.errors.bias_at(t) + self._walk)

    def measure(
        self, t: float, w_body: FloatArray, rng: np.random.Generator
    ) -> tuple[FloatArray, bool]:
        if self.bias_walk_std > 0.0 and self._last_t is not None:
            dt = t - self._last_t
            if dt < 0.0:
                raise ValueError("gyroscope sampled backwards in time")
            self._walk = self._walk + rng.normal(0.0, self.bias_walk_std * math.sqrt(dt), 3)
        self._last_t = t
        x = self.errors.matrix @ w_body + self.true_bias(t)
        return self.errors.finish(x, rng)


_FACE_NORMALS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)


@dataclass(frozen=True)
class SunSensorModel:
    """Sun sensors on the six faces, combined into one Sun-direction reading.

    A face sees the Sun when the angle between its normal and the Sun is within the half
    field of view. The reading is the true direction rotated by Gaussian angular noise. It is
    invalid in eclipse and when no face sees the Sun. Set ``half_fov_rad`` to π/2 or more for
    full-sphere coverage (the thesis simulations assume this).
    """

    noise_std_rad: float = 0.0
    """Angular noise (rad, per transverse axis)."""
    half_fov_rad: float = PSD_HALF_FOV_RAD
    bias: FloatArray = field(default_factory=_vec)
    """Constant additive bias on the unit-vector components."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", _vec(self.bias))

    def visible_faces(self, sun_body: FloatArray) -> NDArray[np.bool_]:
        cos_angle = _FACE_NORMALS @ sun_body
        return np.asarray(cos_angle >= math.cos(min(self.half_fov_rad, math.pi)))

    def measure(
        self, sun_body: FloatArray, eclipse: bool, rng: np.random.Generator
    ) -> tuple[FloatArray, bool, NDArray[np.bool_]]:
        """Returns the reading, its validity and which faces see the Sun."""
        faces = self.visible_faces(sun_body)
        # Draw the noise even when invalid, so the random sequence doesn't depend on eclipses.
        noise = rng.normal(0.0, self.noise_std_rad, 3) if self.noise_std_rad > 0.0 else None
        valid = (not eclipse) and bool(np.any(faces))
        if not valid:
            return np.zeros(3), False, faces & (not eclipse)
        s = sun_body
        if noise is not None:
            s = s + (noise - (noise @ s) * s)  # transverse perturbation only
        return unit(s + self.bias), True, faces
