"""Main window of the viewer (``adcssim view``)."""

import math
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolBar,
    QWidget,
)

from adcssim.launch import RunRequest, prepare
from adcssim.telemetry import Telemetry
from adcssim.viz.groundtrack import GroundTrack
from adcssim.viz.history import History
from adcssim.viz.panels import AdcsPanel, ScenarioPanel, SensorsPanel
from adcssim.viz.recorder import GifRecorder
from adcssim.viz.scene3d import CAMERA_MODES, Scene3D
from adcssim.viz.sources import LiveSource, ReplaySource, Source

SPEEDS = (1, 5, 10, 30, 60, 120, 300, math.inf)
"""Playback speeds. "max" runs the simulation as fast as it can (about 270× on one core)."""
UI_INTERVAL_MS = 33
PLOT_INTERVAL_S = 0.1
"""Redraw the 2D plots at most every 0.1 s of wall time (~10 Hz): they cost far more than the
3D view. Time-based, not every n-th tick, so they keep updating at low speeds, where only some
ticks bring new rows."""
RECORD_EVERY = 3
HISTORY_S = 6000.0
"""Seconds of history kept for plots and tracks: a bit more than one orbit."""


def _fmt_elapsed(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"T+{h:02d}:{m:02d}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self, source: Source, title: str, record_dir: Path = Path("runs")) -> None:
        super().__init__()
        self.source = source
        self.record_dir = record_dir
        self.dt = float(source.metadata.get("dt_adcs_s", "1.0"))
        self.backend = source.metadata.get("backend", "none")
        self.history = History(capacity=max(4, round(HISTORY_S / self.dt)))
        self._ticks = 0
        self._last_plot = -math.inf
        self._prediction_t = -math.inf
        self.recorder: GifRecorder | None = None
        self.setWindowTitle(f"adcssim · {title}")
        self.resize(1600, 950)

        self.scene = Scene3D(self)
        self.ground = GroundTrack()
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.scene.widget)
        left.addWidget(self.ground.widget)
        left.setSizes([620, 300])

        self.sensors = SensorsPanel()
        self.adcs = AdcsPanel()
        self.info = ScenarioPanel()
        self.info.show_metadata(source.metadata)
        tabs = QTabWidget()
        tabs.addTab(self.sensors, "Sensors")
        tabs.addTab(self.adcs, "ADCS")
        tabs.addTab(self.info, "Scenario")
        # Only the visible tab's plots are updated: plotting is the viewer's main cost.
        tabs.currentChanged.connect(lambda _: self.refresh(force_plots=True))
        self.tabs = tabs

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(tabs)
        main.setSizes([1000, 600])
        self.setCentralWidget(main)
        self._build_toolbar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(UI_INTERVAL_MS)

    # ---------------------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("Playback")
        bar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, bar)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_play)
        bar.addWidget(self.play_button)

        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem("max" if math.isinf(s) else f"{s}×", s)
        self.speed_box.setCurrentIndex(SPEEDS.index(60))
        self.speed_box.currentIndexChanged.connect(
            lambda _: self.source.set_speed(float(self.speed_box.currentData()))
        )
        bar.addWidget(QLabel("  speed "))
        bar.addWidget(self.speed_box)

        self.camera_box = QComboBox()
        self.camera_box.addItems(CAMERA_MODES)
        self.camera_box.currentTextChanged.connect(self.scene.set_camera_mode)
        bar.addSeparator()
        bar.addWidget(QLabel(" camera "))
        bar.addWidget(self.camera_box)

        self.record_button = QPushButton("● Record GIF")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.toggle_recording)
        bar.addWidget(self.record_button)
        bar.addSeparator()

        self.slider: QSlider | None = None
        if self.source.seekable:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, max(1, int(self.source.duration_s)))
            self.slider.setMinimumWidth(300)
            self.slider.sliderReleased.connect(self._seek_to_slider)
            bar.addWidget(self.slider)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        self.time_label = QLabel("T+00:00:00   0000-00-00 00:00:00 UTC")
        self.time_label.setStyleSheet("font-family: monospace; padding-right: 8px;")
        self.time_label.setFixedWidth(self.time_label.sizeHint().width())
        bar.addWidget(self.time_label)

    # ---------------------------------------------------------------------------------
    def toggle_play(self) -> None:
        if self.source.playing:
            self.source.pause()
            self.play_button.setText("▶ Play")
        else:
            self.source.set_speed(float(self.speed_box.currentData()))
            self.source.play()
            self.play_button.setText("⏸ Pause")

    def toggle_recording(self, on: bool) -> None:
        if on:
            self.recorder = GifRecorder(self, self.scene)
            self.record_button.setText("■ Stop recording")
            return
        self.record_button.setText("● Record GIF")
        if self.recorder is not None and self.recorder.frames:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            path = self.recorder.save(self.record_dir / f"recording_{stamp}.gif")
            self.statusBar().showMessage(f"Saved {path}", 10_000)
        self.recorder = None

    def _seek_to_slider(self) -> None:
        if self.slider is None:
            return
        rows = self.source.seek(float(self.slider.value()), HISTORY_S)
        self.history.clear()
        self.history.extend(rows)
        self._prediction_t = -math.inf
        self.refresh(force_plots=True)

    # ---------------------------------------------------------------------------------
    def tick(self) -> None:
        error = getattr(self.source, "error", None)
        if error is not None:
            self.timer.stop()
            QMessageBox.critical(self, "Simulation error", str(error))
            return
        rows = self.source.poll()
        if rows:
            self.history.extend(rows)
            self.refresh()
        # Captured on the UI clock, whether or not rows arrived, so a GIF plays back at the
        # speed it was watched.
        if self.recorder is not None and self.source.playing and self._ticks % RECORD_EVERY == 0:
            self.recorder.capture()
        if self.source.finished and self.source.playing:
            self.source.pause()
            self.play_button.setText("▶ Play")
            self.statusBar().showMessage("End of run", 5_000)
        self._ticks += 1

    def refresh(self, force_plots: bool = False) -> None:
        h = self.history
        if len(h) == 0:
            return
        t = float(h.latest("t"))
        self._update_prediction(t)

        start = self.source.clock.start
        utc = (start + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S UTC")
        status = f"{_fmt_elapsed(t)}   {utc}"
        self.time_label.setText(status)
        if self.slider is not None and not self.slider.isSliderDown():
            self.slider.setValue(int(t))

        past = h.last("r", round(5700 / self.dt))[::10] / 1e3
        self.scene.update(
            r_km=h.latest("r") / 1e3,
            q_true=h.latest("q_true"),
            q_est=h.latest("q_est"),
            gmst=float(h.latest("gmst")),
            sun_i=h.latest("sun_i"),
            b_i=h.latest("b_i"),
            eclipse=bool(h.latest("eclipse")),
            past_km=np.vstack([past, h.latest("r") / 1e3]),
            status=status,
            v=h.latest("v"),
        )
        self.scene.render()

        now = time.monotonic()
        if force_plots or now - self._last_plot >= PLOT_INTERVAL_S:
            self._last_plot = now
            # Readouts refresh with the plots: every label change repaints the whole panel.
            self.sensors.update_readouts(h)
            k = round(5700 / self.dt)
            self.ground.update(
                np.degrees(h.last("lat", k)),
                np.degrees(h.last("lon", k)),
                h.latest("sun_i"),
                float(h.latest("gmst")),
            )
            if self.tabs.currentWidget() is self.sensors:
                self.sensors.update_plots(h, self.dt)
            elif self.tabs.currentWidget() is self.adcs:
                self.adcs.refresh(h, self.backend, self.sensors.window_s, self.dt)

    def _update_prediction(self, t: float) -> None:
        """Refresh the next-orbit track when the source has a new one (live: every 60 s of
        simulated time; replay: every 30 s, or after a seek)."""
        if self.source.seekable and 0.0 <= t - self._prediction_t < 30.0:
            return
        prediction = self.source.prediction()
        if prediction is None or float(prediction.t[0]) == self._prediction_t:
            return
        self._prediction_t = float(prediction.t[0])
        self.scene.set_prediction(prediction)
        self.ground.set_prediction(prediction)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self.timer.stop()
        self.source.close()
        self.scene.close()
        super().closeEvent(event)


def open_source(
    path: Path, backend: str | None, speed: float, request: RunRequest | None = None
) -> tuple[Source, str]:
    """A replay source for ``.parquet`` logs, a live simulation for ``.toml`` scenarios.
    ``request`` carries the overrides (variant, seed, …) for a scenario; ``backend``, when
    given, takes precedence over the request's."""
    if path.suffix == ".parquet":
        return ReplaySource(Telemetry.read(path), speed), f"replay of {path.name}"
    request = replace(request or RunRequest(path), scenario=path)
    if backend is not None:
        request = replace(request, backend=backend)
    scenario = prepare(request)
    return (
        LiveSource(scenario, None, speed),
        f"{scenario.name} ({scenario.backend}, {scenario.adcs.variant}, live)",
    )


def open_viewer(
    path: Path,
    speed: float,
    autoplay: bool,
    request: RunRequest | None = None,
    backend: str | None = None,
) -> MainWindow:
    """Open a viewer window (the application must exist). Used by ``adcssim view`` and by
    the launcher, which keeps running while viewers are open."""
    source, title = open_source(path, backend, speed, request)
    window = MainWindow(source, title)
    if math.isinf(speed):
        index = len(SPEEDS) - 1
    else:
        index = min(range(len(SPEEDS) - 1), key=lambda i: abs(SPEEDS[i] - speed))
    window.speed_box.setCurrentIndex(index)
    window.show()
    if autoplay:
        window.toggle_play()
    return window


def run_viewer(
    path: Path,
    backend: str | None,
    speed: float,
    autoplay: bool,
    request: RunRequest | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _window = open_viewer(path, speed, autoplay, request, backend)
    return int(app.exec())
