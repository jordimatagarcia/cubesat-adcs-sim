"""3D view: textured Earth in the inertial (TEME) frame, satellite, orbit and vectors.

Distances are in kilometres. The satellite is drawn hugely enlarged (a real 3U is 34 cm) so
its attitude is visible from orbit scale.
"""

from importlib import resources
from typing import Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from PySide6.QtWidgets import QWidget
from pyvistaqt import QtInteractor

from adcssim.environment import R_EARTH
from adcssim.frames import quat_to_dcm
from adcssim.spacecraft import U3_DIMENSIONS_M
from adcssim.viz.sources import Prediction

FloatArray = NDArray[np.float64]

R_EARTH_KM = R_EARTH / 1e3
SATELLITE_SCALE = 800.0
"""Drawn size of the satellite: km per metre of real size."""
AXIS_LENGTH_KM = 450.0
VECTOR_LENGTH_KM = 1500.0
AXIS_COLOURS = ("#e8453c", "#3cba54", "#4285f4")
CAMERA_MODES = ("track", "follow", "free")
"""track: wide view that keeps the satellite in front; follow: close behind the satellite;
free: the camera stays where the user puts it."""
TRACK_DISTANCE_KM = 30_000.0


def earth_texture_path() -> str:
    return str(resources.files("adcssim").joinpath("data/earth_2048.jpg"))


def earth_mesh(radius: float = R_EARTH_KM, n_lon: int = 180, n_lat: int = 90) -> pv.PolyData:
    """Sphere with equirectangular texture coordinates and a duplicated seam column, so the
    texture doesn't wrap backwards across the ±180° meridian. Coordinates are Earth-fixed."""
    lon = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_g, lat_g = np.meshgrid(lon, lat)  # shape (n_lat + 1, n_lon + 1)
    points = np.column_stack(
        [
            (radius * np.cos(lat_g) * np.cos(lon_g)).ravel(),
            (radius * np.cos(lat_g) * np.sin(lon_g)).ravel(),
            (radius * np.sin(lat_g)).ravel(),
        ]
    )
    cols = n_lon + 1
    i, j = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    a = (i * cols + j).ravel()
    quads = np.column_stack([np.full_like(a, 4), a, a + 1, a + cols + 1, a + cols])
    mesh = pv.PolyData(points, quads.ravel())
    mesh.active_texture_coordinates = np.column_stack(
        [((lon_g + np.pi) / (2 * np.pi)).ravel(), ((lat_g + np.pi / 2) / np.pi).ravel()]
    )
    return mesh.compute_normals(auto_orient_normals=True)


def _lines(points: FloatArray) -> pv.PolyData:
    if len(points) < 2:
        points = np.zeros((2, 3))
    return pv.lines_from_points(points)


def _pose(q: FloatArray, r_km: FloatArray) -> FloatArray:
    """4×4 transform placing body-frame geometry at its inertial attitude and position."""
    m = np.eye(4)
    m[:3, :3] = quat_to_dcm(q).T  # body → inertial
    m[:3, 3] = r_km
    return m


class Scene3D:
    """The PyVista scene and its per-frame updates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.plotter = QtInteractor(parent)
        self.widget: QWidget = self.plotter.interactor
        p = self.plotter
        p.set_background("#05070d")  # type: ignore[arg-type]
        p.remove_all_lights()
        self.sun_light = pv.Light(light_type="scene light")
        self.sun_light.positional = False
        self.sun_light.intensity = 1.1
        p.add_light(self.sun_light)

        self.earth = p.add_mesh(
            earth_mesh(),
            texture=pv.read_texture(earth_texture_path()),
            ambient=0.12,
            diffuse=1.0,
            specular=0.0,
            smooth_shading=True,
        )

        # Satellite (true attitude) and its body axes, posed with one transform.
        dx, dy, dz = (d * SATELLITE_SCALE / 2 for d in U3_DIMENSIONS_M)
        box = pv.Box(bounds=(-dx, dx, -dy, dy, -dz, dz))
        self.sat_actors: list[Any] = [
            p.add_mesh(box, color="#d9d9d9", ambient=0.35, show_edges=True, edge_color="#666666")
        ]
        for axis, colour in zip(np.eye(3), AXIS_COLOURS, strict=True):
            arrow = pv.Arrow(start=(0, 0, 0), direction=axis, scale=AXIS_LENGTH_KM, tip_length=0.2)
            self.sat_actors.append(p.add_mesh(arrow, color=colour, ambient=0.6))
        # Estimated attitude (filled in once an ADCS backend provides one).
        self.ghost = p.add_mesh(box, color="#ffb000", opacity=0.35, ambient=0.6)
        self.ghost.SetVisibility(False)

        self.past_track = _lines(np.zeros((2, 3)))
        self.next_track = _lines(np.zeros((2, 3)))
        p.add_mesh(self.past_track, color="#ffffff", line_width=2, opacity=0.8)
        p.add_mesh(self.next_track, color="#ffd400", line_width=1.5, opacity=0.9)

        self.sun_vec = _lines(np.zeros((2, 3)))
        self.b_vec = _lines(np.zeros((2, 3)))
        p.add_mesh(self.sun_vec, color="#ffcc33", line_width=3)
        p.add_mesh(self.b_vec, color="#d63bd6", line_width=3)

        self.status_text = p.add_text("", position="upper_left", font_size=10, color="white")
        p.add_text(
            "white: past orbit   yellow: next orbit   magenta: B field   orange: Sun",
            position="lower_left",
            font_size=8,
            color="#aaaaaa",
        )
        self.camera_mode = "track"

    def _track_view(self, r_km: FloatArray, v: FloatArray) -> list[tuple[float, ...]]:
        """Wide view from between the orbit normal and the satellite's side of the Earth, so a
        polar orbit isn't seen edge-on and the satellite is never behind the planet."""
        normal = np.cross(r_km, v)
        normal /= np.linalg.norm(normal)
        direction = 0.75 * normal + 0.66 * r_km / np.linalg.norm(r_km)
        direction /= np.linalg.norm(direction)
        up = np.cross(direction, np.cross([0.0, 0.0, 1.0], direction))
        return [tuple(direction * TRACK_DISTANCE_KM), (0.0, 0.0, 0.0), tuple(up)]

    def update(
        self,
        r_km: FloatArray,
        q_true: FloatArray,
        q_est: FloatArray,
        gmst: float,
        sun_i: FloatArray,
        b_i: FloatArray,
        eclipse: bool,
        past_km: FloatArray,
        status: str,
        v: FloatArray | None = None,
    ) -> None:
        self.earth.SetOrientation(0.0, 0.0, float(np.degrees(gmst)))
        self.sun_light.position = tuple(sun_i * 1e6)
        self.sun_light.focal_point = (0.0, 0.0, 0.0)

        pose = _pose(q_true, r_km)
        for actor in self.sat_actors:
            actor.user_matrix = pose
        if np.all(np.isfinite(q_est)):
            self.ghost.user_matrix = _pose(q_est, r_km)
            self.ghost.SetVisibility(True)
        else:
            self.ghost.SetVisibility(False)

        self.past_track.copy_from(_lines(past_km))
        self.sun_vec.copy_from(_lines(np.array([r_km, r_km + sun_i * VECTOR_LENGTH_KM])))
        b_dir = b_i / np.linalg.norm(b_i)
        self.b_vec.copy_from(_lines(np.array([r_km, r_km + b_dir * VECTOR_LENGTH_KM])))
        # add_text with a corner position returns a vtkCornerAnnotation; 2 = upper left.
        self.status_text.SetText(2, status + ("\nECLIPSE" if eclipse else ""))

        if self.camera_mode == "track" and v is not None:
            self.plotter.camera_position = self._track_view(r_km, v)
        elif self.camera_mode == "follow":
            cam = self.plotter.camera
            offset = np.asarray(cam.position) - np.asarray(cam.focal_point)
            cam.focal_point = tuple(r_km)
            cam.position = tuple(r_km + offset)

    def set_prediction(self, prediction: Prediction | None) -> None:
        if prediction is not None:
            self.next_track.copy_from(_lines(prediction.r / 1e3))

    def set_camera_mode(self, mode: str) -> None:
        if mode not in CAMERA_MODES:
            raise ValueError(f"unknown camera mode {mode!r}")
        self.camera_mode = mode
        if mode == "follow":
            # Relative offset; update() then keeps it while moving the focal point.
            self.plotter.camera.position = (0.0, -6_000.0, 2_500.0)
            self.plotter.camera.focal_point = (0.0, 0.0, 0.0)

    def render(self) -> None:
        self.plotter.render()

    def screenshot(self) -> NDArray[np.uint8]:
        return np.asarray(self.plotter.screenshot(return_img=True), dtype=np.uint8)

    def close(self) -> None:
        self.plotter.close()
