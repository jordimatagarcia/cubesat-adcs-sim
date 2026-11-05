"""Records the viewer window to an animated GIF (for the README)."""

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from adcssim.viz.scene3d import Scene3D


def _to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    data = np.ascontiguousarray(rgb[:, :, :3])
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _to_pil(image: QImage) -> Image.Image:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    buf = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(h, image.bytesPerLine())
    return Image.fromarray(buf[:, : 3 * w].reshape(h, w, 3).copy())


class GifRecorder:
    """Grabs the window at each captured frame.

    The VTK view is a native OpenGL window that ``QWidget.grab`` renders black, so its own
    screenshot is painted over that area of the grabbed window.
    """

    def __init__(self, window: QWidget, scene: Scene3D, width: int = 960) -> None:
        self.window = window
        self.scene = scene
        self.width = width
        self.frames: list[Image.Image] = []

    def capture(self) -> None:
        pixmap = self.window.grab()
        view = self.scene.widget
        top_left = view.mapTo(self.window, QPoint(0, 0))
        shot = _to_qimage(self.scene.screenshot())
        painter = QPainter(pixmap)
        painter.drawImage(top_left, shot.scaled(view.width(), view.height()))
        painter.end()
        frame = _to_pil(pixmap.toImage())
        height = round(frame.height * self.width / frame.width)
        self.frames.append(frame.resize((self.width, height), Image.Resampling.LANCZOS))

    def save(self, path: Path, fps: float = 12.0) -> Path:
        if not self.frames:
            raise ValueError("nothing recorded")
        path.parent.mkdir(parents=True, exist_ok=True)
        # One palette for the whole animation and no dithering: per-frame palettes with dither
        # noise compress several times worse.
        reference = self.frames[len(self.frames) // 2].quantize(
            colors=255, method=Image.Quantize.MEDIANCUT
        )
        palette_frames = [
            f.quantize(palette=reference, dither=Image.Dither.NONE) for f in self.frames
        ]
        palette_frames[0].save(
            path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=True,
        )
        self.frames.clear()
        return path
