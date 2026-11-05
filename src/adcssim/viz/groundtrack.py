"""2D ground track: equirectangular map, past and predicted track, day/night terminator."""

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PIL import Image
from PySide6.QtCore import QRectF, Qt

from adcssim.frames import teme_to_ecef
from adcssim.viz.scene3d import earth_texture_path
from adcssim.viz.sources import Prediction

FloatArray = NDArray[np.float64]


def split_at_dateline(lon_deg: FloatArray, lat_deg: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Insert NaN gaps where the track wraps around ±180°, so it isn't drawn across the map."""
    if len(lon_deg) < 2:
        return lon_deg, lat_deg
    jumps = np.nonzero(np.abs(np.diff(lon_deg)) > 180.0)[0] + 1
    return np.insert(lon_deg, jumps, np.nan), np.insert(lat_deg, jumps, np.nan)


def terminator(sun_i: FloatArray, gmst: float) -> tuple[FloatArray, FloatArray, float]:
    """Day/night boundary as latitude per longitude (deg), and the subsolar latitude (deg)."""
    s = teme_to_ecef(sun_i, gmst)
    lat_s = np.arcsin(np.clip(s[2], -1.0, 1.0))
    lon_s = np.arctan2(s[1], s[0])
    lon = np.linspace(-np.pi, np.pi, 361)
    tan_lat = np.tan(lat_s) if abs(lat_s) > 1e-6 else 1e-6
    lat = np.arctan(-np.cos(lon - lon_s) / tan_lat)
    return np.degrees(lon), np.degrees(lat), float(np.degrees(lat_s))


class GroundTrack:
    def __init__(self) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
        self.widget = pg.PlotWidget()
        plot = self.widget.getPlotItem()
        # No aspect lock: in a wide panel it would crop the latitudes. The map stretches.
        plot.setRange(xRange=(-180, 180), yRange=(-90, 90), padding=0)
        plot.setLimits(xMin=-180, xMax=180, yMin=-90, yMax=90)
        plot.setLabel("bottom", "longitude (°)")
        plot.setLabel("left", "latitude (°)")
        plot.showGrid(x=True, y=True, alpha=0.15)

        texture = np.asarray(Image.open(earth_texture_path()).convert("RGB"))
        image = pg.ImageItem(np.flipud(texture))
        image.setRect(QRectF(-180, -90, 360, 180))
        image.setZValue(-100)  # keep the map under the night shading and the tracks
        plot.addItem(image)

        # Night side: the terminator curve, filled towards the pole that is in darkness.
        self._night = plot.plot(
            pen=pg.mkPen(255, 220, 120, 120, width=1), fillBrush=pg.mkBrush(0, 0, 20, 150)
        )
        self._night.setZValue(-50)

        self.past = plot.plot(pen=pg.mkPen("#ffffff", width=2), connect="finite")
        self.future = plot.plot(
            pen=pg.mkPen("#ffd400", width=1.5, style=Qt.PenStyle.DashLine), connect="finite"
        )
        self.marker = pg.ScatterPlotItem(size=11, brush=pg.mkBrush("#e8453c"), pen="w")
        plot.addItem(self.marker)

    def update(
        self,
        lat_deg: FloatArray,
        lon_deg: FloatArray,
        sun_i: FloatArray,
        gmst: float,
    ) -> None:
        x, y = split_at_dateline(lon_deg, lat_deg)
        self.past.setData(x, y)
        if len(lon_deg):
            self.marker.setData([lon_deg[-1]], [lat_deg[-1]])
        lon_t, lat_t, lat_s = terminator(sun_i, gmst)
        pole = -90.0 if lat_s >= 0.0 else 90.0  # the pole tilted away from the Sun is dark
        self._night.setData(lon_t, lat_t)
        self._night.setFillLevel(pole)

    def set_prediction(self, prediction: Prediction | None) -> None:
        if prediction is None:
            self.future.setData([], [])
            return
        x, y = split_at_dateline(np.degrees(prediction.lon), np.degrees(prediction.lat))
        self.future.setData(x, y)
