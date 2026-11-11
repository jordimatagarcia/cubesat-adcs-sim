"""Scenario files (TOML): everything that defines one simulation run.

Values in the file use the units named in each key (``rate_deg_s``, ``noise_std_nT``); the
loaded :class:`Scenario` holds SI units only. Unknown keys are errors, so a typo can't
silently fall back to a default.

Example (see ``scenarios/`` for complete files)::

    name = "case_1a"
    seed = 1

    [time]
    start_offset_s = 0.0      # after the TLE epoch; or start = "2021-10-14T13:00:00Z"
    duration_s = 3000
    dt_dynamics_s = 0.1
    dt_adcs_s = 1.0

    [orbit]
    tle = "../data/tle/aalto-1_2021-287.tle"

    [initial]
    rate_deg_s = [3, 5, -4]

    [sensors.magnetometer]
    noise_std_nT = 1000
"""

import math
import re
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from adcssim.actuators import FS1_MAX_DIPOLE, Magnetorquers
from adcssim.backends.config import AdcsConfig
from adcssim.disturbances import DisturbanceModel
from adcssim.frames import IDENTITY_QUAT, quat_normalize
from adcssim.orbit import Orbit, Tle
from adcssim.sensors import (
    GYRO_LSB_RAD_S,
    GYRO_RANGE_RAD_S,
    MAG_LSB_T,
    MAG_RANGE_T,
    PSD_HALF_FOV_RAD,
    BiasWindow,
    GyroscopeModel,
    MagnetometerModel,
    SensorErrors,
    SunSensorModel,
)
from adcssim.spacecraft import Spacecraft, box_inertia

FloatArray = NDArray[np.float64]
NT = 1e-9


class ScenarioError(ValueError):
    """The scenario file is invalid."""


NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
"""Scenario names: they become part of the log file name, so no path separators."""


class _Table:
    """A TOML table that remembers which keys were read, to reject unknown ones."""

    def __init__(self, data: dict[str, Any], path: str) -> None:
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _where(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        self._used.add(key)
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise ScenarioError(f"missing required key '{self._where(key)}'")
        return self.get(key)

    def number(self, key: str, default: float | None = None) -> float:
        value = self.get(key, default) if default is not None else self.require(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ScenarioError(f"'{self._where(key)}' must be a number")
        if not math.isfinite(value):
            raise ScenarioError(f"'{self._where(key)}' must be finite")
        return float(value)

    def nonnegative(self, key: str, default: float) -> float:
        value = self.number(key, default)
        if value < 0.0:
            raise ScenarioError(f"'{self._where(key)}' must not be negative")
        return value

    def flag(self, key: str, default: bool) -> bool:
        value = self.get(key, default)
        if not isinstance(value, bool):
            raise ScenarioError(f"'{self._where(key)}' must be true or false")
        return value

    def vector(self, key: str, default: tuple[float, ...], size: int = 3) -> FloatArray:
        value = self.get(key, list(default))
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise ScenarioError(f"'{self._where(key)}' must be a list of numbers") from None
        if arr.shape != (size,):
            raise ScenarioError(f"'{self._where(key)}' must have {size} values")
        if not np.all(np.isfinite(arr)):
            raise ScenarioError(f"'{self._where(key)}' must be finite")
        return arr

    def table(self, key: str) -> "_Table":
        value = self.get(key, {})
        if not isinstance(value, dict):
            raise ScenarioError(f"'{self._where(key)}' must be a table")
        return _Table(value, self._where(key))

    def tables(self, key: str) -> list["_Table"]:
        value = self.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise ScenarioError(f"'{self._where(key)}' must be a list of tables")
        return [_Table(v, f"{self._where(key)}[{i}]") for i, v in enumerate(value)]

    def finish(self, children: list["_Table"] | None = None) -> None:
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            raise ScenarioError(f"unknown key(s) {', '.join(self._where(k) for k in unknown)}")
        for child in children or []:
            child.finish()


@dataclass(frozen=True)
class SensorSuite:
    magnetometer: MagnetometerModel
    gyroscope: GyroscopeModel
    sun: SunSensorModel


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    seed: int
    start: datetime
    duration_s: float
    dt_dynamics_s: float
    dt_adcs_s: float
    orbit: Orbit
    spacecraft: Spacecraft
    q0: FloatArray
    w0: FloatArray
    disturbances: DisturbanceModel
    actuators: Magnetorquers
    backend: str
    adcs: AdcsConfig
    sensors_config: dict[str, Any] = field(repr=False)
    """Raw sensor tables; :meth:`make_sensors` builds fresh (stateful) models from them."""
    source: str = field(default="", repr=False)
    """The TOML text the scenario was loaded from."""

    def __post_init__(self) -> None:
        # Checked here, not only when parsing, so overrides (dataclasses.replace, e.g. the
        # CLI's --duration) are checked too.
        if not NAME_PATTERN.fullmatch(self.name):
            raise ScenarioError(
                f"name {self.name!r} must be letters, digits, '_', '-' or '.' (it names the log)"
            )
        times = (self.duration_s, self.dt_dynamics_s, self.dt_adcs_s)
        if not all(math.isfinite(x) and x > 0 for x in times):
            raise ScenarioError("durations and time steps must be positive")
        sub = self.dt_adcs_s / self.dt_dynamics_s
        if sub < 1 - 1e-9 or abs(sub - round(sub)) > 1e-9:
            raise ScenarioError("time.dt_adcs_s must be a whole multiple of time.dt_dynamics_s")
        periods = self.duration_s / self.dt_adcs_s
        if periods < 1 - 1e-9 or abs(periods - round(periods)) > 1e-6:
            raise ScenarioError(
                f"the duration ({self.duration_s:g} s) must be a whole number of control "
                f"periods (time.dt_adcs_s = {self.dt_adcs_s:g} s)"
            )

    @property
    def steps(self) -> int:
        """Number of ADCS control periods in the run."""
        return round(self.duration_s / self.dt_adcs_s)

    @property
    def substeps(self) -> int:
        """Dynamics steps per control period."""
        return round(self.dt_adcs_s / self.dt_dynamics_s)

    def make_sensors(self) -> SensorSuite:
        """New sensor models in their initial state (the gyroscope bias walk is stateful)."""
        return _parse_sensors(_Table(self.sensors_config, "sensors"))


def load_scenario(path: Path) -> Scenario:
    text = path.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioError(f"{path}: {exc}") from None
    try:
        return parse_scenario(data, base_dir=path.parent, source=text)
    except ScenarioError as exc:
        raise ScenarioError(f"{path}: {exc}") from None


def parse_scenario(data: dict[str, Any], base_dir: Path, source: str = "") -> Scenario:
    root = _Table(data, "")
    name = root.require("name")
    description = root.get("description", "")
    seed = root.get("seed", 0)
    adcs_raw = root.get("adcs", {})
    if not isinstance(adcs_raw, dict):
        raise ScenarioError("'adcs' must be a table")
    adcs_raw = dict(adcs_raw)
    backend = adcs_raw.pop("backend", "none")
    try:
        adcs = AdcsConfig.from_table(adcs_raw)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(str(exc)) from None
    if not isinstance(name, str) or not isinstance(description, str):
        raise ScenarioError("'name' and 'description' must be strings")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ScenarioError("'seed' must be a non-negative integer")

    orbit_t = root.table("orbit")
    orbit, epoch = _parse_orbit(orbit_t, base_dir)

    time_t = root.table("time")
    if time_t.has("start"):
        start = _parse_datetime(time_t.get("start"), "time.start")
    else:
        if epoch is None:
            raise ScenarioError("'time.start' is required when the orbit has no epoch")
        start = epoch + timedelta(seconds=time_t.number("start_offset_s", 0.0))
    duration = time_t.number("duration_s")
    dt_dyn = time_t.number("dt_dynamics_s", 0.1)
    dt_adcs = time_t.number("dt_adcs_s", 1.0)

    spacecraft = _parse_spacecraft(root.table("spacecraft"))
    init_t = root.table("initial")
    q_raw = init_t.vector("quaternion", tuple(IDENTITY_QUAT), size=4)
    if not np.linalg.norm(q_raw) > 0.0:
        raise ScenarioError("'initial.quaternion' must not be zero")
    q0 = quat_normalize(q_raw)
    w0 = np.radians(init_t.vector("rate_deg_s", (0.0, 0.0, 0.0)))

    dist_t = root.table("disturbances")
    disturbances = DisturbanceModel(
        gravity_gradient=dist_t.flag("gravity_gradient", True),
        residual_dipole=dist_t.flag("residual_dipole", True),
        aerodynamic=dist_t.flag("aerodynamic", True),
        solar_pressure=dist_t.flag("solar_pressure", True),
    )

    act_t = root.table("actuators")
    levels = act_t.get("levels", 0)
    if not isinstance(levels, int) or isinstance(levels, bool):
        raise ScenarioError("'actuators.levels' must be an integer")
    try:
        actuators = Magnetorquers(
            max_dipole=act_t.vector("max_dipole_A_m2", FS1_MAX_DIPOLE),
            levels=levels,
            duty_cycle=act_t.number("duty_cycle", 1.0),
        )
    except ValueError as exc:
        raise ScenarioError(f"actuators: {exc}") from None

    sensors_raw = root.get("sensors", {})
    if not isinstance(sensors_raw, dict):
        raise ScenarioError("'sensors' must be a table")
    try:
        _parse_sensors(_Table(sensors_raw, "sensors"))  # validate now, build fresh per run
    except ScenarioError:
        raise
    except ValueError as exc:
        raise ScenarioError(f"sensors: {exc}") from None

    root.finish([orbit_t, time_t, init_t, dist_t, act_t])
    if not isinstance(backend, str):
        raise ScenarioError("'adcs.backend' must be a string")

    return Scenario(
        name=name,
        description=description,
        seed=seed,
        start=start,
        duration_s=duration,
        dt_dynamics_s=dt_dyn,
        dt_adcs_s=dt_adcs,
        orbit=orbit,
        spacecraft=spacecraft,
        q0=q0,
        w0=w0,
        disturbances=disturbances,
        actuators=actuators,
        backend=backend,
        adcs=adcs,
        sensors_config=sensors_raw,
        source=source,
    )


def _parse_datetime(value: Any, where: str) -> datetime:
    if isinstance(value, datetime):
        t = value
    elif isinstance(value, str):
        try:
            t = datetime.fromisoformat(value)
        except ValueError:
            raise ScenarioError(f"'{where}' is not an ISO 8601 date-time") from None
    else:
        raise ScenarioError(f"'{where}' must be a date-time")
    if t.tzinfo is None:
        raise ScenarioError(f"'{where}' must include a time zone (e.g. 'Z' for UTC)")
    return t.astimezone(UTC)


def _parse_orbit(t: _Table, base_dir: Path) -> tuple[Orbit, datetime | None]:
    if t.has("tle") == t.has("elements"):
        raise ScenarioError("[orbit] needs exactly one of 'tle' or 'elements'")
    if t.has("tle"):
        rel = t.get("tle")
        if not isinstance(rel, str):
            raise ScenarioError("'orbit.tle' must be a file path")
        try:
            tle = Tle.from_file((base_dir / rel).resolve())
        except (OSError, ValueError) as exc:
            raise ScenarioError(f"orbit.tle: {exc}") from None
        orbit = Orbit.from_tle(tle)
        return orbit, orbit.epoch
    el = t.table("elements")
    epoch = _parse_datetime(el.require("epoch"), "orbit.elements.epoch")
    orbit = Orbit.from_elements(
        epoch,
        altitude_m=el.number("altitude_km") * 1e3,
        inclination_deg=el.number("inclination_deg"),
        raan_deg=el.number("raan_deg"),
        eccentricity=el.number("eccentricity", 0.0),
        arg_perigee_deg=el.number("arg_perigee_deg", 0.0),
        mean_anomaly_deg=el.number("mean_anomaly_deg", 0.0),
        bstar=el.number("bstar", 0.0),
    )
    el.finish()
    return orbit, epoch


def _parse_spacecraft(t: _Table) -> Spacecraft:
    defaults = Spacecraft()
    mass = t.number("mass_kg", defaults.mass_kg)
    if not mass > 0.0:
        raise ScenarioError("'spacecraft.mass_kg' must be positive")
    if t.has("inertia_kg_m2") and t.has("principal_inertia_kg_m2"):
        raise ScenarioError("give either 'inertia_kg_m2' or 'principal_inertia_kg_m2'")
    if t.has("inertia_kg_m2"):
        try:
            inertia = np.asarray(t.get("inertia_kg_m2"), dtype=np.float64)
        except (TypeError, ValueError):
            raise ScenarioError("'spacecraft.inertia_kg_m2' must be a 3x3 matrix") from None
        if not np.all(np.isfinite(inertia)):
            raise ScenarioError("'spacecraft.inertia_kg_m2' must be finite")
    elif t.has("principal_inertia_kg_m2"):
        inertia = np.diag(t.vector("principal_inertia_kg_m2", (0.0, 0.0, 0.0)))
    else:
        # a uniform box of the given mass (the default mass gives the default inertia)
        inertia = box_inertia(mass, defaults.dimensions_m)
    try:
        sc = Spacecraft(
            mass_kg=mass,
            inertia=inertia,
            residual_dipole=t.vector("residual_dipole_A_m2", tuple(defaults.residual_dipole)),
            centre_of_pressure=t.vector("centre_of_pressure_m", tuple(defaults.centre_of_pressure)),
            drag_coefficient=t.nonnegative("drag_coefficient", defaults.drag_coefficient),
            reflectivity_coefficient=t.nonnegative(
                "reflectivity_coefficient", defaults.reflectivity_coefficient
            ),
        )
    except ValueError as exc:
        raise ScenarioError(f"spacecraft: {exc}") from None
    t.finish()
    return sc


def _bias_windows(t: _Table, key: str, scale: float) -> tuple[BiasWindow, ...]:
    windows = []
    for w in t.tables("bias_windows"):
        try:
            windows.append(
                BiasWindow(w.number("start_s"), w.number("end_s"), w.vector(key, ()) * scale)
            )
        except ValueError as exc:
            raise ScenarioError(str(exc)) from None
        w.finish()
    return tuple(windows)


def _errors(t: _Table, unit: str, scale: float, rng_: float, lsb: float) -> SensorErrors:
    quantize = t.flag("quantize", True)
    saturate = t.flag("saturate", True)
    return SensorErrors(
        noise_std=t.nonnegative(f"noise_std_{unit}", 0.0) * scale,
        bias=t.vector(f"bias_{unit}", (0.0, 0.0, 0.0)) * scale,
        bias_windows=_bias_windows(t, f"bias_{unit}", scale),
        scale_factor=t.vector("scale_factor", (0.0, 0.0, 0.0)),
        misalignment=np.radians(t.vector("misalignment_deg", (0.0, 0.0, 0.0))),
        nonorthogonality=np.radians(t.vector("nonorthogonality_deg", (0.0, 0.0, 0.0))),
        range=rng_ if saturate else math.inf,
        lsb=lsb if quantize else 0.0,
    )


def _parse_sensors(t: _Table) -> SensorSuite:
    mag_t = t.table("magnetometer")
    gyro_t = t.table("gyroscope")
    sun_t = t.table("sun")

    mag = MagnetometerModel(
        errors=_errors(mag_t, "nT", NT, MAG_RANGE_T, MAG_LSB_T),
        hard_iron=mag_t.vector("hard_iron_nT", (0.0, 0.0, 0.0)) * NT,
    )
    gyro = GyroscopeModel(
        errors=_errors(gyro_t, "deg_s", math.pi / 180.0, GYRO_RANGE_RAD_S, GYRO_LSB_RAD_S),
        bias_walk_std=math.radians(gyro_t.nonnegative("bias_walk_deg_s_sqrt_s", 0.0)),
    )
    half_fov = sun_t.number("half_fov_deg", math.degrees(PSD_HALF_FOV_RAD))
    if not 0.0 < half_fov <= 180.0:
        raise ScenarioError("'sensors.sun.half_fov_deg' must be in (0, 180]")
    sun = SunSensorModel(
        noise_std_rad=math.radians(sun_t.nonnegative("noise_std_deg", 0.0)),
        half_fov_rad=math.radians(half_fov),
    )
    t.finish([mag_t, gyro_t, sun_t])
    return SensorSuite(mag, gyro, sun)
