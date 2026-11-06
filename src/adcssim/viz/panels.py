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

from adcssim.backends import AdcsMode, AdcsStatus
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


def _sigma3_angle(p_diag: FloatArray) -> FloatArray:
    """3σ bound of the attitude error angle (rad) from the covariance diagonal. The MEKF
    stores the attitude-error variances (rad², indices 0–2, NaN at 3); the thesis filter
    stores quaternion-component variances, and the angle is about twice the vector part."""
    p = np.atleast_2d(p_diag)
    vector_var = p[:, 0] + p[:, 1] + p[:, 2]
    scale = np.where(np.isnan(p[:, 3]), 1.0, 2.0)
    return np.asarray(3.0 * scale * np.sqrt(vector_var))


def _bias_plot(
    title: str, unit: str
) -> tuple[pg.PlotWidget, list[pg.PlotDataItem], list[pg.PlotDataItem], list[pg.PlotDataItem]]:
    """Estimated bias (solid), true bias (dotted) and the estimate's ±3σ (thin dashed)."""
    w, estimate, truth = _plot(title, unit)
    bounds = [_bound_curve(w, c) for c in AXIS_PENS for _ in range(2)]
    return w, estimate, truth, bounds


def _bound_curve(w: pg.PlotWidget, color: str) -> pg.PlotDataItem:
    """A dashed 3σ curve that is left out of the y auto-range: after a fault the filter
    inflates its uncertainty by orders of magnitude, which would flatten everything else."""
    curve = pg.PlotDataItem(pen=pg.mkPen(color, width=0.7, style=pg.QtCore.Qt.PenStyle.DashLine))
    w.getPlotItem().addItem(curve, ignoreBounds=True)
    return curve


def _status_names(status: int) -> str:
    names = [f.name for f in AdcsStatus if f.value and status & f.value and f.name]
    return ", ".join(n.lower().replace("_", " ") for n in names) if names else "nominal"


class AdcsPanel(QWidget):
    """What the ADCS estimates and corrects: attitude error against its 3σ bound, the sensor
    biases it removes from the readings against the true ones, and the commands it sends."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.labels: dict[str, QLabel] = {}
        rows = [
            ("state", "State"),
            ("status", "Status"),
            ("att", "Attitude error"),
            ("mag", "Mag. bias (µT)"),
            ("gyro", "Gyro bias (°/s)"),
        ]
        for i, (key, title) in enumerate(rows):
            grid.addWidget(QLabel(f"<b>{title}</b>"), i, 0)
            label = _readout_label()
            grid.addWidget(label, i, 1)
            self.labels[key] = label
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addWidget(
            QLabel(
                "<small>Solid: estimate · dotted: truth · dashed: estimate ± 3σ. "
                "The ADCS subtracts the estimated biases from the readings.</small>"
            )
        )

        self.err_plot = pg.PlotWidget()
        item = self.err_plot.getPlotItem()
        item.setTitle("Attitude error (estimate vs truth)", size="9pt")
        item.setLabel("left", "°")
        item.showGrid(x=True, y=True, alpha=0.2)
        item.addLegend(offset=(5, 5), labelTextSize="8pt", colCount=2)
        self.err_curve = item.plot(pen=pg.mkPen("#ffb000", width=1.5), name="error")
        self.err_bound = _bound_curve(self.err_plot, "#ffb000")
        item.legend.addItem(self.err_bound, "3σ")
        self.mag_plot, self.mag_est, self.mag_true, self.mag_bounds = _bias_plot(
            "Magnetometer bias", "µT"
        )
        self.gyro_plot, self.gyro_est, self.gyro_true, self.gyro_bounds = _bias_plot(
            "Gyroscope bias", "°/s"
        )
        self.dip_plot, self.dip_curves, _ = _plot("Magnetorquer dipole (applied)", "A·m²")

        self.mode_plot = pg.PlotWidget()
        item = self.mode_plot.getPlotItem()
        item.setLabel("bottom", "t (s)")
        item.showGrid(x=True, y=True, alpha=0.2)
        item.getAxis("left").setTicks([[(m.value, m.name.lower()) for m in AdcsMode]])
        item.setYRange(-0.3, max(AdcsMode) + 0.3, padding=0)
        self.mode_curve = item.plot(pen=pg.mkPen("#cccccc", width=1.5), stepMode=None)
        self.mode_plot.setFixedHeight(140)

        self.plots = (self.err_plot, self.mag_plot, self.gyro_plot, self.dip_plot, self.mode_plot)
        for w in self.plots:
            w.getPlotItem().disableAutoRange(axis="x")
            w.getPlotItem().getAxis("left").setWidth(LEFT_AXIS_WIDTH)
            if w is not self.mode_plot:
                w.getPlotItem().setAutoVisible(y=True)
                w.setMinimumHeight(120)
                layout.addWidget(w, stretch=1)
        layout.addWidget(self.mode_plot)

    def refresh(self, h: History, backend: str, window_s: int, dt: float) -> None:
        self._update_readouts(h, backend)
        k = max(2, round(window_s / dt))
        t = h.last("t", k)
        if len(t) == 0:
            return
        p = h.last("p_diag", k)
        self.err_curve.setData(t, np.degrees(h.last("attitude_error", k)), connect="finite")
        self.err_bound.setData(t, np.degrees(_sigma3_angle(p)), connect="finite")
        for curves, truths, bounds, est, true, sigma in (
            (
                self.mag_est,
                self.mag_true,
                self.mag_bounds,
                h.last("bias_mag_est", k) * 1e6,
                h.last("mag_bias_true", k) * 1e6,
                3.0 * np.sqrt(p[:, 13:16]),  # filter units: µT
            ),
            (
                self.gyro_est,
                self.gyro_true,
                self.gyro_bounds,
                np.degrees(h.last("bias_gyro_est", k)),
                np.degrees(h.last("gyro_bias_true", k)),
                np.degrees(3.0 * np.sqrt(p[:, 16:19])),
            ),
        ):
            for axis in range(3):
                curves[axis].setData(t, est[:, axis], connect="finite")
                truths[axis].setData(t, true[:, axis])
                bounds[2 * axis].setData(t, est[:, axis] + sigma[:, axis], connect="finite")
                bounds[2 * axis + 1].setData(t, est[:, axis] - sigma[:, axis], connect="finite")
        dip = h.last("m_applied", k)
        for axis in range(3):
            self.dip_curves[axis].setData(t, dip[:, axis])
        self.mode_curve.setData(t, h.last("mode", k))
        for plot in self.plots:
            plot.setXRange(float(t[-1]) - window_s, float(t[-1]), padding=0)

    def _update_readouts(self, h: History, backend: str) -> None:
        mode = AdcsMode(int(h.latest("mode")))
        compute = h.latest("compute_time_s") * 1e6
        note = "   (open loop: no ADCS)" if backend == "none" else ""
        self.labels["state"].setText(f"{backend}   {mode.name}   step {compute:.1f} µs{note}")
        self.labels["status"].setText(_status_names(int(h.latest("status"))))
        p = h.latest("p_diag")
        err = math.degrees(float(h.latest("attitude_error")))
        bound = math.degrees(float(_sigma3_angle(p)[0]))
        self.labels["att"].setText(f"{err:.3f}°   (3σ {bound:.3f}°)")
        mag_est = h.latest("bias_mag_est") * 1e6
        mag_true = h.latest("mag_bias_true") * 1e6
        self.labels["mag"].setText(f"{_fmt(mag_est, 1.0, 1)}  (true {_fmt(mag_true, 1.0, 1)})")
        gyro_est = np.degrees(h.latest("bias_gyro_est"))
        gyro_true = np.degrees(h.latest("gyro_bias_true"))
        self.labels["gyro"].setText(f"{_fmt(gyro_est, 1.0, 3)}  (true {_fmt(gyro_true, 1.0, 3)})")


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
