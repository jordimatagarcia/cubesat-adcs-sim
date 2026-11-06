"""Visualization: data plumbing (no display needed) and a GUI smoke test (needs a display)."""

import dataclasses
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyvistaqt", reason="viz extra not installed")

from adcssim.runner import Simulation
from adcssim.scenario import Scenario, parse_scenario
from adcssim.telemetry import Telemetry
from adcssim.viz.groundtrack import split_at_dateline, terminator
from adcssim.viz.history import FIELDS, History
from adcssim.viz.sources import LiveSource, ReplaySource
from tests.conftest import REPO

HAS_DISPLAY = sys.platform != "linux" or bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="no display for OpenGL")


def scenario(duration: float = 40.0) -> Scenario:
    return parse_scenario(
        {
            "name": "viz-test",
            "time": {"duration_s": duration, "start_offset_s": 3816.0},
            "orbit": {"tle": "data/tle/aalto-1_2021-287.tle"},
            "initial": {"rate_deg_s": [3.0, 5.0, -4.0]},
        },
        base_dir=REPO,
    )


@pytest.fixture(scope="module")
def telemetry() -> Telemetry:
    return Simulation(scenario(120.0)).run()


def row(t: float) -> dict[str, Any]:
    return {name: np.full(width, t) if width > 1 else t for name, width in FIELDS.items()}


# --- history --------------------------------------------------------------------------


def test_history_keeps_the_most_recent_rows() -> None:
    h = History(capacity=5)
    for t in range(12):  # wraps the double buffer
        h.append(row(float(t)))
    assert len(h) == 5
    np.testing.assert_array_equal(h.last("t"), [7, 8, 9, 10, 11])
    np.testing.assert_array_equal(h.last("t", 2), [10, 11])
    assert h.last("r").shape == (5, 3)
    assert h.latest("t") == 11
    h.clear()
    assert len(h) == 0
    with pytest.raises(IndexError):
        h.latest("t")


# --- ground track helpers -------------------------------------------------------------


def test_track_is_split_at_the_dateline() -> None:
    lon = np.array([170.0, 175.0, -178.0, -170.0])
    x, _ = split_at_dateline(lon, np.arange(4.0))
    assert np.isnan(x[2]) and len(x) == 5
    np.testing.assert_array_equal(x[[0, 1, 3, 4]], lon)


def test_terminator_at_equinox_noon_on_greenwich() -> None:
    # Sun over (0°, 0°): the terminator runs along the ±90° meridians.
    lon, lat, lat_s = terminator(np.array([1.0, 0.0, 0.0]), 0.0)
    assert lat_s == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.abs(lat[np.abs(np.abs(lon) - 90) > 2]) > 80)


def test_terminator_points_are_ninety_degrees_from_the_sun() -> None:
    sun = np.array([0.6, 0.3, 0.4])
    sun /= np.linalg.norm(sun)
    lon, lat, _ = terminator(sun, 0.0)
    lo, la = np.radians(lon), np.radians(lat)
    p = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    np.testing.assert_allclose(p @ sun, 0.0, atol=1e-9)


# --- sources --------------------------------------------------------------------------


def test_replay_at_max_speed_delivers_every_row(telemetry: Telemetry) -> None:
    src = ReplaySource(telemetry, speed=math.inf)
    src.play()
    rows = src.poll()
    assert len(rows) == len(telemetry)
    assert src.finished
    assert rows[5]["t"] == 5.0 and "gmst" in rows[5]


def test_replay_is_paced_by_speed(telemetry: Telemetry) -> None:
    src = ReplaySource(telemetry, speed=100.0)
    assert src.poll() == []  # paused
    src.play()
    time.sleep(0.2)
    n = len(src.poll())
    assert 10 <= n <= 40  # ~20 simulated seconds in 0.2 s


def test_replay_seek_returns_the_preceding_window(telemetry: Telemetry) -> None:
    src = ReplaySource(telemetry)
    rows = src.seek(60.0, window_s=30.0)
    assert [r["t"] for r in rows] == [float(t) for t in range(30, 61)]
    prediction = src.prediction()
    assert prediction is not None and prediction.t[0] == 60.0


def test_live_source_runs_to_the_end() -> None:
    sc = scenario(25.0)
    src = LiveSource(sc, speed=math.inf)
    try:
        src.play()
        rows: list[dict[str, Any]] = []
        deadline = time.monotonic() + 60.0
        while not src.finished and time.monotonic() < deadline:
            rows += src.poll()
            time.sleep(0.01)
        rows += src.poll()
        assert src.error is None
        assert [r["t"] for r in rows] == [float(t) for t in range(25)]
        prediction = src.prediction()
        assert prediction is not None
        assert prediction.t[-1] - prediction.t[0] >= 0.95 * sc.orbit.period_s
    finally:
        src.close()


# --- GUI ------------------------------------------------------------------------------


@needs_display
def test_window_live_and_recording(qtbot: Any, tmp_path: Path) -> None:
    from adcssim.viz.app import MainWindow

    window = MainWindow(LiveSource(scenario(60.0), speed=math.inf), "test", record_dir=tmp_path)
    qtbot.addWidget(window)
    window.show()
    window.record_button.setChecked(True)
    window.toggle_play()
    qtbot.waitUntil(lambda: len(window.history) >= 30, timeout=30_000)
    qtbot.waitUntil(
        lambda: window.recorder is not None and len(window.recorder.frames) > 0, timeout=5_000
    )
    window.record_button.setChecked(False)
    gifs = list(tmp_path.glob("recording_*.gif"))
    assert len(gifs) == 1 and gifs[0].stat().st_size > 0
    window.close()


@needs_display
def test_window_replay_seek(qtbot: Any, telemetry: Telemetry) -> None:
    from adcssim.viz.app import MainWindow

    window = MainWindow(ReplaySource(telemetry), "replay")
    qtbot.addWidget(window)
    window.show()
    assert window.slider is not None
    window.slider.setValue(90)
    window._seek_to_slider()
    assert window.history.latest("t") == 90.0
    assert "T+00:01:30" in window.time_label.text()
    window.close()


@needs_display
def test_adcs_tab_shows_estimates_and_biases(qtbot: Any) -> None:
    from adcssim.viz.app import MainWindow

    sc = dataclasses.replace(scenario(60.0), backend="python-ref")
    window = MainWindow(ReplaySource(Simulation(sc).run()), "python-ref")
    qtbot.addWidget(window)
    window.show()
    window.tabs.setCurrentWidget(window.adcs)
    assert window.slider is not None
    window.slider.setValue(50)
    window._seek_to_slider()
    panel = window.adcs
    assert "ESTIMATE" in panel.labels["state"].text()
    assert "3σ" in panel.labels["att"].text()
    t, err = panel.err_curve.getData()
    assert t is not None and len(t) == 51 and np.nanmax(err) < 5.0
    t, bound = panel.err_bound.getData()
    assert np.all(np.isfinite(bound[-10:])) and np.all(bound[-10:] > 0.0)
    _, gyro_bias = panel.gyro_est[2].getData()
    assert np.all(np.isfinite(gyro_bias[-10:]))
    window.close()


def test_live_source_reports_errors_from_the_process() -> None:
    with pytest.raises(ValueError, match="unknown ADCS backend"):
        LiveSource(scenario(5.0), backend="quantum")


def test_live_source_close_stops_the_process() -> None:
    src = LiveSource(scenario(3000.0), speed=10.0)
    src.play()
    time.sleep(0.5)
    src.close()
    assert not src._proc.is_alive()
