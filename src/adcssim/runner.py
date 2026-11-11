"""The simulation loop: truth → sensors → ADCS → actuators → dynamics.

Each call to :meth:`Simulation.step` runs one ADCS control period:

1. Sample the sensors at the start of the period and call the ADCS backend.
2. Hold its magnetorquer command (zero-order hold, with the duty cycle) and integrate the
   attitude over the period in fixed RK4 steps. The environment (position, field, Sun) is
   evaluated at every dynamics step; attitude-dependent torques are re-evaluated inside each
   RK4 stage.
"""

import dataclasses
import json
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adcssim import __version__
from adcssim.actuators import magnetic_torque
from adcssim.backends import AdcsBackend, AdcsContext, AdcsInput, AdcsOutput, create_backend
from adcssim.backends.native import LIBRARIES, library_path
from adcssim.dynamics import rk4_step
from adcssim.environment import Environment, EnvironmentSample
from adcssim.frames import quat_angle, quat_to_dcm
from adcssim.igrf import Igrf
from adcssim.scenario import Scenario
from adcssim.telemetry import DIAGNOSTICS, SCHEMA, Telemetry, TelemetryRecorder
from adcssim.timebase import Clock

FloatArray = NDArray[np.float64]


ENV_BATCH_PERIODS = 64
"""Control periods of environment evaluated per batch (see ``Simulation._environment``)."""


@cache
def _igrf() -> Igrf:
    return Igrf()


@dataclass(frozen=True)
class Frame:
    """Everything known at the start of one control period (truth, readings, ADCS output)."""

    t: float
    env: EnvironmentSample
    """Environment at ``t`` (a batch of one)."""
    q_true: FloatArray
    w_true: FloatArray
    b_b_true: FloatArray
    sun_b_true: FloatArray
    adcs_in: AdcsInput
    adcs_out: AdcsOutput
    sun_faces: NDArray[np.bool_]
    mag_bias_true: FloatArray
    gyro_bias_true: FloatArray
    m_applied: FloatArray
    tau_control: FloatArray
    tau_disturbance: FloatArray
    compute_time_s: float

    @property
    def attitude_error(self) -> float:
        """Angle between estimated and true attitude (rad), NaN without an estimate."""
        if not np.all(np.isfinite(self.adcs_out.q_est)):
            return float("nan")
        return float(quat_angle(self.adcs_out.q_est, self.q_true))

    def telemetry_row(self) -> dict[str, object]:
        e, i, o = self.env, self.adcs_in, self.adcs_out
        return {
            "t": self.t,
            "r": e.r[0],
            "v": e.v[0],
            "lat": e.lat[0],
            "lon": e.lon[0],
            "alt": e.alt[0],
            "eclipse": e.eclipse[0],
            "b_i": e.b[0],
            "sun_i": e.sun[0],
            "q_true": self.q_true,
            "w_true": self.w_true,
            "b_b_true": self.b_b_true,
            "sun_b_true": self.sun_b_true,
            "mag_bias_true": self.mag_bias_true,
            "gyro_bias_true": self.gyro_bias_true,
            "mag": i.mag_b,
            "mag_valid": i.mag_valid,
            "gyro": i.gyro_b,
            "gyro_valid": i.gyro_valid,
            "sun": i.sun_b,
            "sun_valid": i.sun_valid,
            "sun_faces": int(np.packbits(self.sun_faces, bitorder="little")[0]),
            "q_est": o.q_est,
            "w_est": o.w_est,
            "bias_mag_est": o.bias_mag,
            "bias_gyro_est": o.bias_gyro,
            "p_diag": o.p_diag,
            "m_cmd": o.m_cmd,
            "mode": int(o.mode),
            "status": o.status,
            "compute_time_s": self.compute_time_s,
            "m_applied": self.m_applied,
            "tau_control": self.tau_control,
            "tau_disturbance": self.tau_disturbance,
            "attitude_error": self.attitude_error,
            **self._diagnostics(),
        }

    def _diagnostics(self) -> dict[str, object]:
        diag = self.adcs_out.diag or {}
        out: dict[str, object] = {}
        for name in DIAGNOSTICS:
            width = SCHEMA[name][0]
            out[name] = diag.get(name, np.nan if width == 1 else np.full(width, np.nan))
        return out


class Simulation:
    """One run of a scenario with an ADCS backend."""

    def __init__(self, scenario: Scenario, backend: AdcsBackend | None = None) -> None:
        self.scenario = scenario
        self.clock = Clock(scenario.start)
        self.env = Environment(scenario.orbit, self.clock, _igrf())
        self.sensors = scenario.make_sensors()
        rng_mag, rng_gyro, rng_sun = np.random.SeedSequence(scenario.seed).spawn(3)
        self._rng_mag = np.random.default_rng(rng_mag)
        self._rng_gyro = np.random.default_rng(rng_gyro)
        self._rng_sun = np.random.default_rng(rng_sun)
        self.backend = backend if backend is not None else create_backend(scenario.backend)
        self.backend.reset(
            AdcsContext(
                dt=scenario.dt_adcs_s,
                inertia=scenario.spacecraft.inertia.copy(),
                max_dipole=scenario.actuators.max_dipole.copy(),
                config=scenario.adcs,
            )
        )
        self.q = scenario.q0.copy()
        self.w = scenario.w0.copy()
        self.k = 0
        self._inertia_inv = np.linalg.inv(scenario.spacecraft.inertia)
        self._env_batch: EnvironmentSample | None = None
        self._env_first = -1
        lo, hi = self.env.igrf.valid_range
        years = (self.clock.decimal_year(0.0), self.clock.decimal_year(scenario.duration_s))
        if not (lo <= years[0] and years[1] <= hi):
            raise ValueError(
                f"the run ({years[0]:.2f}–{years[1]:.2f}) is outside the IGRF model's range "
                f"({lo:.1f}–{hi:.1f}); change time.start"
            )

    @property
    def t(self) -> float:
        return self.k * self.scenario.dt_adcs_s

    @property
    def done(self) -> bool:
        return self.k >= self.scenario.steps

    def step(self) -> Frame:
        """Run one control period and return the frame recorded at its start."""
        if self.done:
            raise RuntimeError("simulation already finished")
        sc = self.scenario
        dt = sc.dt_dynamics_s
        t = self.t
        env = self._environment(self.k)
        env0 = env.at(0)

        a = quat_to_dcm(self.q)
        b_b = a @ env.b[0]
        sun_b = a @ env.sun[0]
        eclipse = bool(env.eclipse[0])
        mag, mag_ok = self.sensors.magnetometer.measure(t, b_b, self._rng_mag)
        gyro, gyro_ok = self.sensors.gyroscope.measure(t, self.w, self._rng_gyro)
        sun, sun_ok, faces = self.sensors.sun.measure(sun_b, eclipse, self._rng_sun)
        adcs_in = AdcsInput(
            t=t,
            mag_b=mag,
            mag_valid=mag_ok,
            gyro_b=gyro,
            gyro_valid=gyro_ok,
            sun_b=sun,
            sun_valid=sun_ok,
            sun_ref_i=env.sun[0].copy(),
            mag_ref_i=env.b[0].copy(),
        )
        tic = time.perf_counter_ns()
        adcs_out = self.backend.step(adcs_in)
        compute_time = (time.perf_counter_ns() - tic) * 1e-9
        m_applied = sc.actuators.apply(adcs_out.m_cmd)

        coils_on_at_start = sc.actuators.active(0.0, sc.dt_adcs_s)
        tau_control = magnetic_torque(m_applied, b_b) if coils_on_at_start else np.zeros(3)
        tau_disturbance = sc.disturbances.torque(
            sc.spacecraft,
            a @ env.r[0],
            b_b,
            a @ env.v_rel[0],
            float(env.density[0]),
            sun_b,
            float(env.sun_distance[0]),
            eclipse,
        )
        frame = Frame(
            t=t,
            env=env0,
            q_true=self.q.copy(),
            w_true=self.w.copy(),
            b_b_true=b_b,
            sun_b_true=sun_b,
            adcs_in=adcs_in,
            adcs_out=adcs_out,
            sun_faces=faces,
            mag_bias_true=self.sensors.magnetometer.true_bias(t),
            gyro_bias_true=self.sensors.gyroscope.true_bias(t),
            m_applied=m_applied,
            tau_control=tau_control,
            tau_disturbance=tau_disturbance,
            compute_time_s=compute_time,
        )

        # The coils switch off at t_off into the period. A substep that straddles t_off is
        # split there, so the applied impulse is exact for any duty cycle, not rounded to whole
        # substeps. The environment is held at the substep's start either way.
        t_off = sc.actuators.on_time(sc.dt_adcs_s)
        off = np.zeros(3)
        for j in range(sc.substeps):
            t0 = j * dt
            on_part = min(max(t_off - t0, 0.0), dt)
            if on_part < 1e-12 * dt:
                pieces = [(dt, off)]
            elif dt - on_part < 1e-12 * dt:
                pieces = [(dt, m_applied)]
            else:
                pieces = [(on_part, m_applied), (dt - on_part, off)]
            for h, m in pieces:
                self.q, self.w = rk4_step(
                    self.q,
                    self.w,
                    h,
                    sc.spacecraft.inertia,
                    self._torque_fn(env, j, m),
                    self._inertia_inv,
                )
        if not (np.all(np.isfinite(self.q)) and np.all(np.isfinite(self.w))):
            raise FloatingPointError(
                f"the true attitude became non-finite at t = {t + sc.dt_adcs_s:g} s "
                f"(|w| was {float(np.linalg.norm(frame.w_true)):.3g} rad/s at t = {t:g} s)"
            )
        self.k += 1
        return frame

    def _environment(self, k: int) -> EnvironmentSample:
        """Environment at the substep starts of control period ``k``.

        The environment does not depend on the attitude, so it is evaluated for
        ``ENV_BATCH_PERIODS`` periods at a time (the IGRF sum is the costly part, and its
        per-call overhead dominates for a single period's handful of points). Batches are
        aligned to absolute period numbers, so the result does not depend on where a run was
        resumed or stepped from.
        """
        sc = self.scenario
        n = sc.substeps
        first = (k // ENV_BATCH_PERIODS) * ENV_BATCH_PERIODS
        if self._env_batch is None or self._env_first != first:
            last = min(first + ENV_BATCH_PERIODS, sc.steps)
            idx = np.arange(first * n, last * n)
            self._env_batch = self.env.evaluate(idx * sc.dt_dynamics_s)
            self._env_first = first
        start = (k - first) * n
        return self._env_batch.slice(start, start + n)

    def _torque_fn(
        self, env: EnvironmentSample, j: int, m: FloatArray
    ) -> Callable[[FloatArray, FloatArray], FloatArray]:
        sc = self.scenario
        b_i, r_i, v_i, sun_i = env.b[j], env.r[j], env.v_rel[j], env.sun[j]
        density, sun_dist, eclipse = (
            float(env.density[j]),
            float(env.sun_distance[j]),
            bool(env.eclipse[j]),
        )

        def torque(q: FloatArray, w: FloatArray) -> FloatArray:
            a = quat_to_dcm(q)
            b_b = a @ b_i
            tau = magnetic_torque(m, b_b)
            return tau + sc.disturbances.torque(
                sc.spacecraft, a @ r_i, b_b, a @ v_i, density, a @ sun_i, sun_dist, eclipse
            )

        return torque

    def frames(self) -> Iterator[Frame]:
        """Iterate over the remaining control periods."""
        while not self.done:
            yield self.step()

    def run(self, progress: Callable[[int, int], None] | None = None) -> Telemetry:
        """Run to the end and return the telemetry log."""
        recorder = TelemetryRecorder(self.scenario.steps - self.k)
        for frame in self.frames():
            recorder.append(frame.telemetry_row())
            if progress is not None:
                progress(self.k, self.scenario.steps)
        return recorder.finish(self.metadata())

    def metadata(self) -> dict[str, str]:
        return run_metadata(self.scenario, self.backend.name)


def run_metadata(sc: Scenario, backend_name: str) -> dict[str, str]:
    """Metadata stored with a run's telemetry."""
    return {
        "adcssim_version": __version__,
        "scenario_name": sc.name,
        "scenario_toml": sc.source,
        "backend": backend_name,
        "adcs_variant": sc.adcs.variant,
        "seed": str(sc.seed),
        "duration_s": repr(sc.duration_s),
        "start_utc": sc.start.isoformat(),
        "dt_adcs_s": repr(sc.dt_adcs_s),
        "dt_dynamics_s": repr(sc.dt_dynamics_s),
        "adcs_config": json.dumps(dataclasses.asdict(sc.adcs), sort_keys=True, default=_jsonable),
        "git_commit": _git_commit(),
        "library": _library(backend_name),
    }


@cache
def _git_commit() -> str:
    """Commit of the source tree the simulator runs from ("+dirty" if it has local changes),
    or "" when it is not a git checkout (e.g. an installed wheel)."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return head + ("+dirty" if dirty else "")


def _library(backend_name: str) -> str:
    """Path of the compiled library a native backend loads, or "" for the Python backend."""
    if backend_name not in LIBRARIES:
        return ""
    try:
        return str(library_path(backend_name).resolve())
    except ValueError:
        return ""


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
