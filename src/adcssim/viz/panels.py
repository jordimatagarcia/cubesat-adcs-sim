"""Side panels: sensor readings, ADCS state and scenario information."""

import html
import math

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from adcssim.backends import AdcsMode
from adcssim.viz.history import History

FloatArray = NDArray[np.float64]

AXIS_PENS = ("#e8453c", "#3cba54", "#4285f4")
LEFT_AXIS_WIDTH = 60
"""Fixed y-axis width in pixels. With automatic widths, changing tick labels resize the plot,
which re-lays out and repaints the whole window on every update."""
FACES = ("+X", "−X", "+Y", "−Y", "+Z", "−Z")
WINDOWS_S = (60, 300, 900, 3600)


def _plot(
    title: str, unit: str
) -> tuple[pg.PlotWidget, list[pg.PlotDataItem], list[pg.PlotDataItem]]:
    """A rolling plot with x/y/z reading curves and faint truth curves."""
    w = pg.PlotWidget()
    item = w.getPlotItem()
    item.setTitle(title, size="9pt")
    item.setLabel("left", unit)
    item.setLabel("bottom", "t (s)")
    item.showGrid(x=True, y=True, alpha=0.2)
    item.addLegend(offset=(5, 5), labelTextSize="8pt", colCount=3)
    readings = [
        item.plot(pen=pg.mkPen(c, width=1.2), name=a) for c, a in zip(AXIS_PENS, "xyz", strict=True)
    ]
    truth = [
        item.plot(pen=pg.mkPen(c, width=1, style=pg.QtCore.Qt.PenStyle.DotLine)) for c in AXIS_PENS
    ]
    # The time axis is set explicitly on each update: auto-ranging x on every redraw is the
    # most expensive part of a scrolling plot.
    item.disableAutoRange(axis="x")
    item.setAutoVisible(y=True)
    item.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
    w.setMinimumHeight(150)
    return w, readings, truth


def _readout_label() -> QLabel:
    """A label whose changing text never changes its layout size. Otherwise every update
    re-lays out the panel and forces all the plots below it to repaint."""
    label = QLabel("–")
    label.setStyleSheet("font-family: monospace;")
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    label.setFixedHeight(label.sizeHint().height())
    return label


def _fmt(v: FloatArray, scale: float = 1.0, digits: int = 1) -> str:
    return "  ".join(f"{x * scale:+.{digits}f}" for x in v)


class SensorsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        self.labels: dict[str, QLabel] = {}
        rows = [
            ("mag", "Magnetometer (nT)"),
            ("gyro", "Gyroscope (°/s)"),
            ("sun", "Sun vector"),
            ("faces", "Faces seeing the Sun"),
            ("pos", "Position"),
        ]
        for i, (key, title) in enumerate(rows):
            grid.addWidget(QLabel(f"<b>{title}</b>"), i, 0)
            label = _readout_label()
            grid.addWidget(label, i, 1)
            self.labels[key] = label
        layout.addLayout(grid)

        controls = QHBoxLayout()
        self.truth_box = QCheckBox("show truth (dotted)")
        self.window_box = QComboBox()
        for s in WINDOWS_S:
            self.window_box.addItem(f"last {s // 60} min" if s >= 60 else f"last {s} s", s)
        self.window_box.setCurrentIndex(1)
        controls.addWidget(self.truth_box)
        controls.addStretch()
        controls.addWidget(self.window_box)
        layout.addLayout(controls)

        self.mag_plot, self.mag_curves, self.mag_truth = _plot("Magnetometer", "nT")
        self.gyro_plot, self.gyro_curves, self.gyro_truth = _plot("Gyroscope", "°/s")
        self.sun_plot, self.sun_curves, self.sun_truth = _plot("Sun sensor (unit vector)", "")
        for w in (self.mag_plot, self.gyro_plot, self.sun_plot):
            layout.addWidget(w, stretch=1)

    @property
    def window_s(self) -> int:
        return int(self.window_box.currentData())

    def update_readouts(self, h: History) -> None:
        mag, gyro, sun = h.latest("mag"), h.latest("gyro"), h.latest("sun")
        ok = lambda flag: "" if flag else "  <span style='color:#e8453c'>INVALID</span>"  # noqa: E731
        self.labels["mag"].setText(
            f"{_fmt(mag, 1e9, 0)}   |B| {np.linalg.norm(mag) * 1e9:.0f}{ok(h.latest('mag_valid'))}"
        )
        self.labels["gyro"].setText(
            f"{_fmt(np.degrees(gyro), 1.0, 3)}   |ω| {math.degrees(np.linalg.norm(gyro)):.2f}"
            f"{ok(h.latest('gyro_valid'))}"
        )
        sun_valid = bool(h.latest("sun_valid"))
        eclipse = bool(h.latest("eclipse"))
        self.labels["sun"].setText(
            (_fmt(sun, 1.0, 3) if sun_valid else "no reading")
            + ("  (eclipse)" if eclipse else ok(sun_valid))
        )
        faces = int(h.latest("sun_faces"))
        self.labels["faces"].setText(
            " ".join(
                f"<span style='color:{'#ffd400' if faces >> i & 1 else '#555'}'>{f}</span>"
                for i, f in enumerate(FACES)
            )
        )
        lat, lon = math.degrees(h.latest("lat")), math.degrees(h.latest("lon"))
        self.labels["pos"].setText(
            f"lat {lat:+6.2f}°  lon {lon:+7.2f}°  alt {h.latest('alt') / 1e3:.1f} km"
        )

    def update_plots(self, h: History, dt: float) -> None:
        k = max(2, round(self.window_s / dt))
        t = h.last("t", k)
        show_truth = self.truth_box.isChecked()
        sun = h.last("sun", k).copy()
        sun[~h.last("sun_valid", k).astype(bool)] = np.nan
        series = [
            (self.mag_curves, self.mag_truth, h.last("mag", k) * 1e9, h.last("b_b_true", k) * 1e9),
            (
                self.gyro_curves,
                self.gyro_truth,
                np.degrees(h.last("gyro", k)),
                np.degrees(h.last("w_true", k)),
            ),
            (self.sun_curves, self.sun_truth, sun, h.last("sun_b_true", k)),
        ]
        if len(t) == 0:
            return
        x_range = (float(t[-1]) - self.window_s, float(t[-1]))
        for plot, (curves, truths, values, truth) in zip(
            (self.mag_plot, self.gyro_plot, self.sun_plot), series, strict=True
        ):
            for axis in range(3):
                curves[axis].setData(t, values[:, axis], connect="finite")
                if show_truth:
                    truths[axis].setData(t, truth[:, axis])
                else:
                    truths[axis].setData([], [])
            plot.setXRange(*x_range, padding=0)


class AdcsPanel(QWidget):
    """ADCS state. Phase 5 extends this with raw-vs-corrected sensor views."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.state = _readout_label()
        self.state.setFixedHeight(2 * self.state.height())
        layout.addWidget(self.state)

        self.err_plot = pg.PlotWidget()
        item = self.err_plot.getPlotItem()
        item.setTitle("Attitude error (estimate vs truth)", size="9pt")
        item.setLabel("left", "°")
        item.setLabel("bottom", "t (s)")
        item.showGrid(x=True, y=True, alpha=0.2)
        self.err_curve = item.plot(pen=pg.mkPen("#ffb000", width=1.5))
        item.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        item.disableAutoRange(axis="x")
        item.setAutoVisible(y=True)
        layout.addWidget(self.err_plot, stretch=1)

        self.dip_plot = pg.PlotWidget()
        item = self.dip_plot.getPlotItem()
        item.setTitle("Magnetorquer dipole (applied)", size="9pt")
        item.setLabel("left", "A·m²")
        item.setLabel("bottom", "t (s)")
        item.showGrid(x=True, y=True, alpha=0.2)
        self.dip_curves = [item.plot(pen=pg.mkPen(c, width=1.2)) for c in AXIS_PENS]
        item.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        item.disableAutoRange(axis="x")
        item.setAutoVisible(y=True)
        layout.addWidget(self.dip_plot, stretch=1)

    def refresh(self, h: History, backend: str, window_s: int, dt: float) -> None:
        mode = AdcsMode(int(h.latest("mode")))
        compute = h.latest("compute_time_s") * 1e6
        note = "open loop: no ADCS backend" if backend == "none" else ""
        self.state.setText(
            f"backend {backend}   mode {mode.name}   status 0x{int(h.latest('status')):08x}\n"
            f"step time {compute:.1f} µs   {note}"
        )
        k = max(2, round(window_s / dt))
        t = h.last("t", k)
        self.err_curve.setData(t, np.degrees(h.last("attitude_error", k)), connect="finite")
        dip = h.last("m_applied", k)
        for axis in range(3):
            self.dip_curves[axis].setData(t, dip[:, axis])
        if len(t):
            for plot in (self.err_plot, self.dip_plot):
                plot.setXRange(float(t[-1]) - window_s, float(t[-1]), padding=0)


class ScenarioPanel(QTextBrowser):
    def show_metadata(self, meta: dict[str, str]) -> None:
        rows = "".join(
            f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(meta.get(k, '–'))}</td></tr>"
            for k in (
                "scenario_name",
                "backend",
                "seed",
                "start_utc",
                "dt_adcs_s",
                "dt_dynamics_s",
                "adcssim_version",
            )
        )
        toml = html.escape(meta.get("scenario_toml", ""))
        self.setHtml(f"<table cellpadding='3'>{rows}</table><hr><pre>{toml}</pre>")
