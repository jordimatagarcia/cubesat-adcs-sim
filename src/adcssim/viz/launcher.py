"""Launcher window (``adcssim gui``): choose a scenario, the ADCS implementation (Python,
C, Rust or none), the algorithm variant and the run options, then open a live view, run
to a log, or replay a log.

All the logic lives in :mod:`adcssim.launch`; this module is only the form. Every choice
maps onto a :class:`~adcssim.launch.RunRequest`, whose equivalent command line is shown at
the bottom of the window.
"""

import math
import shlex
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QFont, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from adcssim.backends.abi import VARIANT_IDS
from adcssim.backends.config import VARIANTS
from adcssim.launch import (
    RUNS_DIR,
    SCENARIO_DIR,
    SECTIONS,
    VARIANT_LABELS,
    AdcsField,
    BatchResult,
    RunCancelledError,
    RunRequest,
    ScenarioInfo,
    adcs_fields,
    backend_status,
    format_adcs_value,
    list_scenarios,
    parse_adcs_value,
    prepare,
    recent_logs,
    run_batch,
)
from adcssim.scenario import Scenario, ScenarioError, load_scenario
from adcssim.viz.app import SPEEDS, MainWindow, open_viewer

PATH_ROLE = Qt.ItemDataRole.UserRole
SECTION_FOR_VARIANT = {
    "thesis": ("FLAE", "Two-stage UKF (thesis)", "Control"),
    "improved": ("FLAE", "Error-state filters (improved, MEKF)", "Control"),
    "mekf": ("FLAE", "Error-state filters (improved, MEKF)", "Control"),
}
"""Sections of the advanced settings expanded for each variant (the others still apply
to nothing, so they stay collapsed)."""


class _BatchWorker(QObject):
    """Runs :func:`run_batch` on a worker thread and reports back through signals."""

    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, scenario: Scenario, out_dir: Path) -> None:
        super().__init__()
        self.scenario = scenario
        self.out_dir = out_dir
        self.cancel = threading.Event()

    def run(self) -> None:
        every = max(1, self.scenario.steps // 200)

        def progress(k: int, n: int) -> None:
            if k % every == 0 or k == n:
                self.progress.emit(k, n)

        try:
            result = run_batch(self.scenario, self.out_dir, progress, self.cancel.is_set)
        except RunCancelledError:
            self.failed.emit("Cancelled; no log was written.")
        except Exception as exc:  # reported in the window, not lost
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)


class LauncherWindow(QMainWindow):
    def __init__(self, scenario_dir: Path = SCENARIO_DIR) -> None:
        super().__init__()
        self.setWindowTitle("adcssim · launcher")
        self.resize(1200, 820)
        self.settings = QSettings("adcssim", "launcher")
        self.fields = adcs_fields()
        self.scenario: Scenario | None = None
        self.viewers: list[MainWindow] = []
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None
        self._ready = False  # signals fired while the form is being built are ignored

        left = QWidget()
        lcol = QVBoxLayout(left)
        lcol.addWidget(self._build_scenarios(scenario_dir))
        lcol.addWidget(self._build_implementation())
        lcol.addWidget(self._build_options())

        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.addWidget(self._build_advanced(), stretch=3)
        rcol.addWidget(self._build_recent(), stretch=1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([560, 640])

        central = QWidget()
        col = QVBoxLayout(central)
        col.addWidget(split, stretch=1)
        col.addWidget(self._build_actions(), stretch=0)
        self.setCentralWidget(central)

        self._ready = True
        self._restore()
        self._variant_changed()

    # ---- building the form ------------------------------------------------------------
    def _build_scenarios(self, scenario_dir: Path) -> QGroupBox:
        box = QGroupBox("Scenario")
        lay = QVBoxLayout(box)
        self.scenario_list = QListWidget()
        self.scenario_list.setMinimumHeight(190)
        for info in list_scenarios(scenario_dir):
            self._add_scenario_item(info)
        self.scenario_list.currentItemChanged.connect(lambda *_: self._scenario_changed())
        lay.addWidget(self.scenario_list)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_scenario)
        lay.addWidget(browse, alignment=Qt.AlignmentFlag.AlignLeft)
        self.scenario_info = QLabel()
        self.scenario_info.setWordWrap(True)
        self.scenario_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.scenario_info)
        return box

    def _add_scenario_item(self, info: ScenarioInfo) -> QListWidgetItem:
        if info.error:
            text = f"{info.path.name}  (does not load)"
        else:
            text = f"{info.name}  ·  {info.duration_s:g} s  ·  {info.backend}, {info.variant}"
        item = QListWidgetItem(text)
        item.setData(PATH_ROLE, str(info.path))
        item.setToolTip(info.error or info.description)
        self.scenario_list.addItem(item)
        return item

    def _build_implementation(self) -> QGroupBox:
        box = QGroupBox("ADCS implementation and algorithm")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.backend_group = QButtonGroup(box)
        self.backend_buttons: dict[str, QRadioButton] = {}
        for status in backend_status():
            button = QRadioButton(status.label)
            button.setProperty("backend", status.name)
            self.backend_group.addButton(button)
            self.backend_buttons[status.name] = button
            row.addWidget(button)
        self.backend_group.buttonToggled.connect(lambda *_: self._backend_changed())
        recheck = QPushButton("Re-check")
        recheck.setToolTip("Check again whether the C and Rust libraries are built")
        recheck.clicked.connect(self._refresh_backends)
        row.addWidget(recheck)
        row.addStretch()
        form.addRow("Implementation", row)
        self.backend_note = QLabel()
        self.backend_note.setWordWrap(True)
        self.backend_note.setStyleSheet("color: #a05a00;")
        form.addRow(self.backend_note)
        self._refresh_backends()

        self.variant_box = QComboBox()
        for v in VARIANTS:
            self.variant_box.addItem(VARIANT_LABELS[v], v)
        self.variant_box.currentIndexChanged.connect(lambda _: self._variant_changed())
        form.addRow("Algorithm", self.variant_box)
        self._sync_variants()
        return box

    def _build_options(self) -> QGroupBox:
        box = QGroupBox("Run options (empty = the scenario's value)")
        form = QFormLayout(box)
        self.duration_edit = QLineEdit()
        self.seed_edit = QLineEdit()
        for edit in (self.duration_edit, self.seed_edit):
            edit.textChanged.connect(lambda _: self._update_command())
        form.addRow("Duration (s)", self.duration_edit)
        form.addRow("Random seed", self.seed_edit)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(str(RUNS_DIR))
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(32)
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_browse)
        form.addRow("Log folder", out_row)

        live_row = QHBoxLayout()
        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem("max" if math.isinf(s) else f"{s}×", s)
        self.speed_box.setCurrentIndex(SPEEDS.index(60))
        self.paused_check = QCheckBox("start paused")
        live_row.addWidget(self.speed_box)
        live_row.addWidget(self.paused_check)
        live_row.addStretch()
        form.addRow("Live view speed", live_row)
        return box

    def _build_advanced(self) -> QGroupBox:
        box = QGroupBox("Advanced ADCS settings")
        lay = QVBoxLayout(box)
        hint = QLabel(
            "Double-click a value to change it. Bold: differs from the scenario. "
            "Vectors: numbers separated by commas."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Setting", "Value", "Scenario value"])
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 200)
        self.section_items: dict[str, QTreeWidgetItem] = {}
        self.field_items: dict[str, QTreeWidgetItem] = {}
        for section in SECTIONS:
            parent = QTreeWidgetItem([section])
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(parent)
            self.section_items[section] = parent
        for f in self.fields:
            item = QTreeWidgetItem([f.name, "", ""])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(0, f.doc or f.name)
            self.section_items[f.section].addChild(item)
            self.field_items[f.name] = item
        self.tree.itemDoubleClicked.connect(self._edit_item)
        self.tree.itemChanged.connect(lambda item, _col: self._setting_changed(item))
        lay.addWidget(self.tree)
        reset = QPushButton("Reset to the scenario's values")
        reset.clicked.connect(self._reset_settings)
        lay.addWidget(reset, alignment=Qt.AlignmentFlag.AlignLeft)
        return box

    def _build_recent(self) -> QGroupBox:
        box = QGroupBox("Recent logs (double-click to replay)")
        lay = QVBoxLayout(box)
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self._replay(Path(item.data(PATH_ROLE)))
        )
        lay.addWidget(self.recent_list)
        self._refresh_recent()
        return box

    def _build_actions(self) -> QWidget:
        w = QWidget()
        col = QVBoxLayout(w)
        self.command_label = QLabel()
        self.command_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.command_label.setFont(mono)
        self.command_label.setWordWrap(True)
        col.addWidget(self.command_label)

        row = QHBoxLayout()
        self.live_button = QPushButton("▶ Open live view")
        self.batch_button = QPushButton("⏵ Run to log")
        self.replay_button = QPushButton("Replay a log…")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.live_button.clicked.connect(self._open_live)
        self.batch_button.clicked.connect(self._start_batch)
        self.replay_button.clicked.connect(self._browse_replay)
        self.cancel_button.clicked.connect(self._cancel_batch)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        for b in (self.live_button, self.batch_button, self.replay_button):
            row.addWidget(b)
        row.addWidget(self.progress, stretch=1)
        row.addWidget(self.cancel_button)
        col.addLayout(row)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono)
        self.output.setFixedHeight(110)
        col.addWidget(self.output)
        return w

    # ---- state -------------------------------------------------------------------------
    def selected_path(self) -> Path | None:
        item = self.scenario_list.currentItem()
        return Path(item.data(PATH_ROLE)) if item is not None else None

    def selected_backend(self) -> str:
        button = self.backend_group.checkedButton()
        return str(button.property("backend")) if button is not None else "python-ref"

    def selected_variant(self) -> str:
        return str(self.variant_box.currentData())

    def adcs_overrides(self) -> dict[str, Any]:
        """Settings whose text differs from the scenario's. Raises ``ValueError``."""
        out: dict[str, Any] = {}
        if self.scenario is None:
            return out
        for f in self.fields:
            text = self.field_items[f.name].text(1)
            if text != self._scenario_text(f):
                out[f.name] = parse_adcs_value(f, text)
        return out

    def request(self) -> RunRequest:
        """The run the form describes. Raises ``ValueError`` on a field that does not
        parse."""
        path = self.selected_path()
        if path is None:
            raise ValueError("choose a scenario first")
        duration = self.duration_edit.text().strip()
        seed = self.seed_edit.text().strip()
        try:
            duration_s = float(duration) if duration else None
        except ValueError:
            raise ValueError(f"duration: {duration!r} is not a number") from None
        try:
            seed_i = int(seed) if seed else None
        except ValueError:
            raise ValueError(f"seed: {seed!r} is not an integer") from None
        return RunRequest(
            scenario=path,
            backend=self.selected_backend(),
            variant=self.selected_variant(),
            duration_s=duration_s,
            seed=seed_i,
            adcs=self.adcs_overrides(),
        )

    def _scenario_text(self, f: AdcsField) -> str:
        assert self.scenario is not None
        return format_adcs_value(getattr(self.scenario.adcs, f.name))

    # ---- reactions ---------------------------------------------------------------------
    def _scenario_changed(self) -> None:
        path = self.selected_path()
        self.scenario = None
        if path is not None:
            try:
                self.scenario = load_scenario(path)
            except (OSError, ScenarioError) as exc:
                self.scenario_info.setText(f"<span style='color:#b00020'>{exc}</span>")
        if self.scenario is not None:
            sc = self.scenario
            self.scenario_info.setText(
                f"<b>{sc.name}</b>: {sc.description}<br>"
                f"{sc.duration_s:g} s from {sc.start:%Y-%m-%d %H:%M} UTC, ADCS every "
                f"{sc.dt_adcs_s:g} s, seed {sc.seed}; scenario default: {sc.backend}, "
                f"{sc.adcs.variant}"
            )
            self.duration_edit.setPlaceholderText(f"{sc.duration_s:g}")
            self.seed_edit.setPlaceholderText(str(sc.seed))
        self._reset_settings()
        for button in (self.live_button, self.batch_button):
            button.setEnabled(self.scenario is not None)

    def _backend_changed(self) -> None:
        self._sync_variants()
        self._update_command()

    def _sync_variants(self) -> None:
        """Grey out the algorithms the selected implementation lacks: the C and Rust
        libraries implement the thesis variant and the MEKF, not the UKF ("improved"). If an
        unavailable one is selected, switch to the MEKF (same model, same settings)."""
        if not hasattr(self, "variant_box"):
            return
        backend = self.selected_backend()
        model = self.variant_box.model()
        assert isinstance(model, QStandardItemModel)
        for i in range(self.variant_box.count()):
            variant = str(self.variant_box.itemData(i))
            available = backend not in ("c", "rust") or variant in VARIANT_IDS
            item = model.item(i)
            item.setEnabled(available)
            item.setToolTip(
                ""
                if available
                else "Only in the Python reference: the C and Rust libraries "
                "implement the thesis variant and the MEKF"
            )
        if self.selected_variant() not in VARIANT_IDS and backend in ("c", "rust"):
            self.variant_box.setCurrentIndex(self.variant_box.findData("mekf"))

    def _variant_changed(self) -> None:
        if not self._ready:
            return
        wanted = SECTION_FOR_VARIANT[self.selected_variant()]
        for section, item in self.section_items.items():
            item.setExpanded(section in wanted)
        self._update_command()

    def _reset_settings(self) -> None:
        self.tree.blockSignals(True)
        for f in self.fields:
            item = self.field_items[f.name]
            text = self._scenario_text(f) if self.scenario is not None else ""
            item.setText(1, text)
            item.setText(2, text)
            self._mark(item, changed=False)
        self.tree.blockSignals(False)
        self._variant_changed()

    def _edit_item(self, item: QTreeWidgetItem, column: int) -> None:
        if item.parent() is not None and column != 1:
            self.tree.editItem(item, 1)

    def _setting_changed(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None or self.scenario is None:
            return
        f = next(f for f in self.fields if f.name == item.text(0))
        self._mark(item, changed=item.text(1) != self._scenario_text(f))
        self._update_command()

    @staticmethod
    def _mark(item: QTreeWidgetItem, changed: bool) -> None:
        font = item.font(1)
        font.setBold(changed)
        item.setFont(1, font)
        item.setFont(0, font)

    def _refresh_backends(self) -> None:
        notes, reasons = [], []
        for status in backend_status():
            button = self.backend_buttons[status.name]
            button.setEnabled(status.available)
            button.setToolTip(status.reason or status.label)
            if not status.available:
                notes.append(f"{status.label} not built: {status.build_command}")
                reasons.append(status.reason)
                if button.isChecked():
                    self.backend_buttons["python-ref"].setChecked(True)
        self.backend_note.setText("\n".join(notes))
        self.backend_note.setToolTip("\n".join(reasons))  # the full path that was looked for
        self.backend_note.setVisible(bool(notes))

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        for path in recent_logs(Path(self.out_edit.text() or RUNS_DIR)):
            item = QListWidgetItem(path.name)
            item.setData(PATH_ROLE, str(path))
            self.recent_list.addItem(item)

    def _update_command(self) -> None:
        if not self._ready:
            return
        try:
            req = self.request()
        except ValueError as exc:
            self.command_label.setText(f"⚠ {exc}")
            return
        if req.scenario.is_relative_to(Path.cwd()):  # shorter, and what one would type
            req = replace(req, scenario=req.scenario.relative_to(Path.cwd()))
        args = req.command_line("run")
        args[2] = Path(args[2]).as_posix()  # forward slashes work in every shell
        line = shlex.join(args)
        if req.adcs:
            changed = ", ".join(sorted(req.adcs))
            line += f"\n# plus ADCS settings with no command-line form: {changed}"
        self.command_label.setText(line)

    def _say(self, text: str, error: bool = False) -> None:
        self.output.appendPlainText(("error: " if error else "") + text)

    # ---- actions -----------------------------------------------------------------------
    def _prepared(self) -> Scenario | None:
        try:
            return prepare(self.request())
        except (ScenarioError, ValueError) as exc:
            self._say(str(exc), error=True)
            return None

    def _open_live(self) -> None:
        if self._prepared() is None:  # report invalid input here, not in the viewer
            return
        path = self.selected_path()
        assert path is not None
        try:
            window = open_viewer(
                path,
                float(self.speed_box.currentData()),
                autoplay=not self.paused_check.isChecked(),
                request=self.request(),
            )
        except (ScenarioError, ValueError, KeyError) as exc:
            self._say(str(exc), error=True)
            return
        self._keep(window)
        self._say(f"opened a live view: {window.windowTitle()}")

    def _replay(self, path: Path) -> None:
        try:
            window = open_viewer(path, float(self.speed_box.currentData()), autoplay=False)
        except (OSError, ValueError, KeyError) as exc:
            self._say(f"{path}: {exc}", error=True)
            return
        self._keep(window)

    def _keep(self, window: MainWindow) -> None:
        self.viewers.append(window)
        window.destroyed.connect(lambda *_: self._forget(window))
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def _forget(self, window: MainWindow) -> None:
        if window in self.viewers:
            self.viewers.remove(window)

    def _start_batch(self) -> None:
        scenario = self._prepared()
        if scenario is None:
            return
        out_dir = Path(self.out_edit.text() or RUNS_DIR)
        self._worker = _BatchWorker(scenario, out_dir)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._batch_progress)
        self._worker.finished.connect(self._batch_done)
        self._worker.failed.connect(self._batch_failed)
        self._set_running(True)
        self.progress.setRange(0, scenario.steps)
        self.progress.setValue(0)
        self._say(f"running {scenario.name} on {scenario.backend} ({scenario.adcs.variant})…")
        self._thread.start()

    def _cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel.set()

    def _batch_progress(self, k: int, n: int) -> None:
        self.progress.setValue(k)
        self.progress.setFormat(f"{k}/{n} steps (%p%)")

    def _batch_done(self, result: BatchResult) -> None:
        self._say("\n".join(result.summary()))
        self._finish_batch()
        self._refresh_recent()

    def _batch_failed(self, message: str) -> None:
        self._say(message, error=not message.startswith("Cancelled"))
        self._finish_batch()

    def _finish_batch(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        for b in (self.live_button, self.batch_button):
            b.setEnabled(not running and self.scenario is not None)
        self.cancel_button.setEnabled(running)

    # ---- file dialogs ------------------------------------------------------------------
    def _browse_scenario(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open scenario", str(SCENARIO_DIR), "Scenarios (*.toml)"
        )
        if name:
            self._select_scenario(Path(name))

    def _select_scenario(self, path: Path) -> None:
        for i in range(self.scenario_list.count()):
            item = self.scenario_list.item(i)
            if Path(item.data(PATH_ROLE)).resolve() == path.resolve():
                self.scenario_list.setCurrentItem(item)
                return
        [info] = [i for i in list_scenarios(path.parent) if i.path.name == path.name]
        self.scenario_list.setCurrentItem(self._add_scenario_item(info))

    def _browse_out(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Log folder", self.out_edit.text())
        if name:
            self.out_edit.setText(name)
            self._refresh_recent()

    def _browse_replay(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Replay a log", self.out_edit.text(), "Telemetry logs (*.parquet)"
        )
        if name:
            self._replay(Path(name))

    # ---- remembered choices ------------------------------------------------------------
    def _restore(self) -> None:
        s = self.settings
        backend = str(s.value("backend", "python-ref"))
        button = self.backend_buttons.get(backend)
        if button is None or not button.isEnabled():
            button = self.backend_buttons["python-ref"]
        button.setChecked(True)
        variant = str(s.value("variant", "thesis"))
        self.variant_box.setCurrentIndex(max(0, self.variant_box.findData(variant)))
        self.out_edit.setText(str(s.value("out_dir", str(RUNS_DIR))))
        self.speed_box.setCurrentIndex(int(str(s.value("speed_index", SPEEDS.index(60)))))
        self.paused_check.setChecked(str(s.value("paused", "false")) == "true")
        last = s.value("scenario")
        if last and Path(str(last)).is_file():
            self._select_scenario(Path(str(last)))
        elif self.scenario_list.count():
            self.scenario_list.setCurrentRow(0)

    def _save(self) -> None:
        s = self.settings
        path = self.selected_path()
        if path is not None:
            s.setValue("scenario", str(path))
        s.setValue("backend", self.selected_backend())
        s.setValue("variant", self.selected_variant())
        s.setValue("out_dir", self.out_edit.text())
        s.setValue("speed_index", self.speed_box.currentIndex())
        s.setValue("paused", "true" if self.paused_check.isChecked() else "false")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self._cancel_batch()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._save()
        for window in list(self.viewers):
            window.close()
        super().closeEvent(event)


def run_launcher() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    return int(app.exec())
