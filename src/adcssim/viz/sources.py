"""Where the viewer gets its rows from: a live simulation or a recorded log.

Both sources produce telemetry rows (the dictionaries of :meth:`Frame.telemetry_row`) plus
``gmst``, paced by a playback speed (simulated seconds per wall-clock second).
"""

import math
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from adcssim.backends import create_backend
from adcssim.frames import ecef_to_geodetic, teme_to_ecef
from adcssim.orbit import MU_EARTH_WGS72
from adcssim.runner import Simulation, run_metadata
from adcssim.scenario import Scenario
from adcssim.telemetry import SCHEMA, Telemetry
from adcssim.timebase import Clock, gmst
from adcssim.viz.history import Row

FloatArray = NDArray[np.float64]

PREDICTION_STEP_S = 30.0
PREDICTION_REFRESH_S = 60.0


@dataclass(frozen=True)
class Prediction:
    """Predicted positions over the next orbit."""

    t: FloatArray
    r: FloatArray
    """TEME positions (m), shape (N, 3)."""
    lat: FloatArray
    """Geodetic latitude (rad)."""
    lon: FloatArray
    """Longitude (rad)."""


def _prediction(clock: Clock, t: FloatArray, r: FloatArray) -> Prediction:
    lat, lon, _ = ecef_to_geodetic(teme_to_ecef(r, gmst(*clock.jd(t))))
    return Prediction(t, r, lat, lon)


class Source(Protocol):
    metadata: dict[str, str]
    clock: Clock
    duration_s: float
    seekable: bool

    @property
    def playing(self) -> bool: ...
    @property
    def finished(self) -> bool: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def set_speed(self, speed: float) -> None: ...
    def poll(self) -> list[Row]: ...
    def seek(self, t: float, window_s: float) -> list[Row]: ...
    def prediction(self) -> Prediction | None: ...
    def close(self) -> None: ...


class _Pacer:
    """Converts wall-clock time into a simulation-time target at a given speed."""

    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.playing = False
        self._sim_ref = 0.0
        self._wall_ref = 0.0

    def start(self, sim_now: float) -> None:
        self._sim_ref, self._wall_ref = sim_now, time.monotonic()
        self.playing = True

    def stop(self) -> None:
        self.playing = False

    def set_speed(self, speed: float, sim_now: float) -> None:
        self.speed = speed
        if self.playing:
            self.start(sim_now)

    def target(self) -> float:
        if not self.playing:
            return -math.inf
        if math.isinf(self.speed):
            return math.inf
        return self._sim_ref + self.speed * (time.monotonic() - self._wall_ref)


class LiveSource:
    """Runs a :class:`Simulation` in a child process, paced by the playback speed.

    A process rather than a thread: the simulation is CPU-bound Python, and in the same
    process it would compete with the UI for the interpreter lock (at 300× the UI dropped to
    about one frame per second). Rows come back in batches through a queue; play, pause and
    speed commands go through a pipe.
    """

    seekable = False

    def __init__(self, scenario: Scenario, backend: str | None = None, speed: float = 60.0) -> None:
        backend_name = backend or scenario.backend
        create_backend(backend_name)  # fail early, in this process, on an unknown name
        self.scenario = scenario
        self.clock = Clock(scenario.start)
        self.duration_s = scenario.duration_s
        self.metadata = run_metadata(scenario, backend_name)
        self._playing = False
        self._done = False
        self._error: BaseException | None = None
        self._prediction: Prediction | None = None
        self._pending: list[Row] = []
        ctx = mp.get_context("spawn")  # never fork a process that runs Qt
        self._out: Queue[tuple[str, Any]] = ctx.Queue()
        self._commands, child_end = ctx.Pipe()
        self._proc = ctx.Process(
            target=_simulation_process,
            args=(scenario, backend_name, speed, child_end, self._out),
            name="adcssim-simulation",
            daemon=True,
        )
        self._proc.start()

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def finished(self) -> bool:
        self._drain()
        return self._done and not self._pending

    @property
    def error(self) -> BaseException | None:
        self._drain()
        return self._error

    def _send(self, *command: Any) -> None:
        if self._proc.is_alive():
            self._commands.send(command)

    def play(self) -> None:
        self._playing = True
        self._send("play")

    def pause(self) -> None:
        self._playing = False
        self._send("pause")

    def set_speed(self, speed: float) -> None:
        self._send("speed", speed)

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self._out.get_nowait()
            except queue.Empty:
                return
            if kind == "rows":
                self._pending.extend(payload)
            elif kind == "prediction":
                self._prediction = payload
            elif kind == "done":
                self._done = True
            elif kind == "error":
                self._error = RuntimeError(payload)
                self._done = True

    def poll(self) -> list[Row]:
        self._drain()
        rows, self._pending = self._pending, []
        return rows

    def seek(self, t: float, window_s: float) -> list[Row]:
        raise NotImplementedError("a live simulation can't seek")

    def prediction(self) -> Prediction | None:
        self._drain()
        return self._prediction

    def close(self) -> None:
        self._send("stop")
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._out.close()
        self._commands.close()


BATCH_INTERVAL_S = 0.02
"""How often the simulation process sends its new rows to the viewer."""


def _simulation_process(
    scenario: Scenario,
    backend_name: str,
    speed: float,
    commands: Connection,
    out: "Queue[tuple[str, Any]]",
) -> None:
    """Entry point of the simulation process."""
    try:
        sim = Simulation(scenario, create_backend(backend_name))
        clock = sim.clock
        period = scenario.orbit.period_s
        pacer = _Pacer(speed)
        next_prediction = 0.0
        batch: list[Row] = []
        last_flush = time.monotonic()
        done_sent = False
        while True:
            while commands.poll():
                command = commands.recv()
                if command[0] == "stop":
                    return
                if command[0] == "play":
                    pacer.start(sim.t)
                elif command[0] == "pause":
                    pacer.stop()
                elif command[0] == "speed":
                    pacer.set_speed(float(command[1]), sim.t)
            due = not sim.done and sim.t <= pacer.target()
            now = time.monotonic()
            if batch and (not due or now - last_flush >= BATCH_INTERVAL_S):
                out.put(("rows", batch))
                batch, last_flush = [], now
            if not due:
                if sim.done and not done_sent:
                    out.put(("done", None))
                    done_sent = True
                time.sleep(0.005)
                continue
            if sim.t >= next_prediction:
                t, r = scenario.orbit.predict(clock, sim.t, period, PREDICTION_STEP_S)
                out.put(("prediction", _prediction(clock, t, r)))
                next_prediction = sim.t + PREDICTION_REFRESH_S
            frame = sim.step()
            row = frame.telemetry_row()
            row["gmst"] = float(gmst(*clock.jd(frame.t)))
            batch.append(row)
    except Exception:
        out.put(("error", traceback.format_exc()))


class ReplaySource:
    """Plays back a telemetry log written by ``adcssim run``."""

    seekable = True

    def __init__(self, telemetry: Telemetry, speed: float = 60.0) -> None:
        self.metadata = dict(telemetry.metadata)
        start = datetime.fromisoformat(self.metadata["start_utc"])
        self.clock = Clock(start)
        self._cols = {name: telemetry.get(name) for name in SCHEMA}
        self._t = np.asarray(self._cols["t"], dtype=np.float64)
        self._cols["gmst"] = gmst(*self.clock.jd(self._t))
        self._dt = float(self._t[1] - self._t[0]) if len(self._t) > 1 else 1.0
        self.duration_s = float(self._t[-1] + self._dt) if len(self._t) else 0.0
        radius = float(np.mean(np.linalg.norm(self._cols["r"], axis=1)))
        self._period = 2.0 * math.pi * math.sqrt(radius**3 / MU_EARTH_WGS72)
        self._pacer = _Pacer(speed)
        self._next = 0
        """Index of the next row to deliver."""

    @property
    def playing(self) -> bool:
        return self._pacer.playing

    @property
    def finished(self) -> bool:
        return self._next >= len(self._t)

    @property
    def position(self) -> float:
        """Simulation time of the next row."""
        return float(self._next * self._dt)

    def _row(self, i: int) -> Row:
        return {name: col[i] for name, col in self._cols.items()}

    def play(self) -> None:
        self._pacer.start(self.position)

    def pause(self) -> None:
        self._pacer.stop()

    def set_speed(self, speed: float) -> None:
        self._pacer.set_speed(speed, self.position)

    def poll(self) -> list[Row]:
        target = self._pacer.target()
        if target == -math.inf:  # paused
            return []
        end = len(self._t) if target == math.inf else int(np.searchsorted(self._t, target, "right"))
        end = min(end, len(self._t))
        rows = [self._row(i) for i in range(self._next, end)]
        self._next = max(self._next, end)
        return rows

    def seek(self, t: float, window_s: float) -> list[Row]:
        """Jump to time ``t``; returns the rows of the preceding window to refill plots."""
        i = int(np.clip(np.searchsorted(self._t, t, "right"), 0, len(self._t)))
        first = int(np.searchsorted(self._t, t - window_s, "left"))
        self._next = i
        if self._pacer.playing:
            self._pacer.start(self.position)
        return [self._row(j) for j in range(first, i)]

    def prediction(self) -> Prediction | None:
        """Positions over the next orbit, taken from the log itself."""
        i = max(self._next - 1, 0)
        step = max(1, round(PREDICTION_STEP_S / self._dt))
        stop = min(len(self._t), i + round(self._period / self._dt) + 1)
        idx = np.arange(i, stop, step)
        if len(idx) < 2:
            return None
        t = self._t[idx]
        lat = np.asarray(self._cols["lat"][idx], dtype=np.float64)
        lon = np.asarray(self._cols["lon"][idx], dtype=np.float64)
        return Prediction(t, np.asarray(self._cols["r"][idx], dtype=np.float64), lat, lon)

    def close(self) -> None:
        pass
