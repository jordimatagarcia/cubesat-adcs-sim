"""Launcher window (``adcssim gui``): the form matches what adcssim.launch reports, and its
buttons run what the form describes. The logic itself is tested in test_launch.py."""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyvistaqt", reason="viz extra not installed")

from PySide6.QtCore import QSettings

import adcssim.viz.launcher as launcher_module
from adcssim.launch import backend_status, list_scenarios
from adcssim.telemetry import Telemetry
from adcssim.viz.launcher import LauncherWindow
from tests.conftest import REPO

HAS_DISPLAY = sys.platform != "linux" or bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="no display")

CASE_1A = REPO / "scenarios" / "case_1a.toml"


@pytest.fixture
def window(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LauncherWindow:
    """A launcher with its own settings store (the user's remembered choices are not read
    or overwritten) and logs going to ``tmp_path``."""
    store = QSettings(str(tmp_path / "launcher.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(launcher_module, "QSettings", lambda *_: store)
    w = LauncherWindow()
    qtbot.addWidget(w)
    w.out_edit.setText(str(tmp_path / "runs"))
    w.scenario_list.setCurrentRow(0)
    return w


def select(w: LauncherWindow, path: Path) -> None:
    w._select_scenario(path)
    assert w.selected_path() == path


def test_form_lists_scenarios_and_backends(window: LauncherWindow) -> None:
    assert window.scenario_list.count() == len(list_scenarios())
    for status in backend_status():
        assert window.backend_buttons[status.name].isEnabled() == status.available
    select(window, CASE_1A)
    assert window.duration_edit.placeholderText() == "3000"
    assert "Slow tumbling" in window.scenario_info.text()


def test_missing_library_is_greyed_out_with_the_build_command(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADCSSIM_RUST_LIB", str(tmp_path / "missing.dll"))
    store = QSettings(str(tmp_path / "launcher.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(launcher_module, "QSettings", lambda *_: store)
    w = LauncherWindow()
    qtbot.addWidget(w)
    assert not w.backend_buttons["rust"].isEnabled()
    assert "cargo build --release" in w.backend_note.text()
    assert "missing.dll" in w.backend_note.toolTip()


def test_request_follows_the_form(window: LauncherWindow) -> None:
    select(window, CASE_1A)
    window.backend_buttons["none"].setChecked(True)
    window.variant_box.setCurrentIndex(window.variant_box.findData("improved"))
    window.duration_edit.setText("12")
    window.seed_edit.setText("3")
    window.field_items["sigma_gyro_noise"].setText(1, "0.002")
    req = window.request()
    assert (req.backend, req.variant, req.duration_s, req.seed) == ("none", "improved", 12.0, 3)
    assert dict(req.adcs) == {"sigma_gyro_noise": 0.002}
    assert window.field_items["sigma_gyro_noise"].font(1).bold()
    assert "--variant improved" in window.command_label.text()
    assert "sigma_gyro_noise" in window.command_label.text()

    window._reset_settings()
    assert dict(window.request().adcs) == {}
    assert not window.field_items["sigma_gyro_noise"].font(1).bold()


def test_variant_expands_its_settings(window: LauncherWindow) -> None:
    window.variant_box.setCurrentIndex(window.variant_box.findData("mekf"))
    assert window.section_items["Error-state filters (improved, MEKF)"].isExpanded()
    assert not window.section_items["Two-stage UKF (thesis)"].isExpanded()
    window.variant_box.setCurrentIndex(window.variant_box.findData("thesis"))
    assert window.section_items["Two-stage UKF (thesis)"].isExpanded()


def test_invalid_input_is_reported_and_nothing_runs(window: LauncherWindow) -> None:
    window.duration_edit.setText("abc")
    assert "not a number" in window.command_label.text()
    window.batch_button.click()
    assert "error: duration" in window.output.toPlainText()
    assert window._thread is None


def test_run_to_log(qtbot: Any, window: LauncherWindow, tmp_path: Path) -> None:
    select(window, CASE_1A)
    window.backend_buttons["python-ref"].setChecked(True)
    window.variant_box.setCurrentIndex(window.variant_box.findData("mekf"))
    window.duration_edit.setText("2")
    window.seed_edit.setText("6")
    window.batch_button.click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60_000)
    [log] = (tmp_path / "runs").glob("*.parquet")
    meta = Telemetry.read(log).metadata
    assert (meta["adcs_variant"], meta["seed"]) == ("mekf", "6")
    assert "att. error" in window.output.toPlainText()
    assert window.recent_list.count() == 1
    assert window.batch_button.isEnabled() and not window.cancel_button.isEnabled()


def test_cancel_stops_a_run_without_a_log(
    qtbot: Any, window: LauncherWindow, tmp_path: Path
) -> None:
    select(window, CASE_1A)
    window.backend_buttons["python-ref"].setChecked(True)
    window.batch_button.click()  # the full 3000 s
    qtbot.waitUntil(lambda: window.progress.value() > 10, timeout=60_000)
    window.cancel_button.click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60_000)
    assert "Cancelled" in window.output.toPlainText()
    assert not list((tmp_path / "runs").glob("*.parquet"))


def test_choices_are_remembered(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = QSettings(str(tmp_path / "launcher.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(launcher_module, "QSettings", lambda *_: store)
    first = LauncherWindow()
    qtbot.addWidget(first)
    select(first, REPO / "scenarios" / "case_2a.toml")
    first.variant_box.setCurrentIndex(first.variant_box.findData("improved"))
    first.paused_check.setChecked(True)
    first.close()

    second = LauncherWindow()
    qtbot.addWidget(second)
    assert second.selected_path() == REPO / "scenarios" / "case_2a.toml"
    assert second.selected_variant() == "improved"
    assert second.paused_check.isChecked()
