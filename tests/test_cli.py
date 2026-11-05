from pathlib import Path

import pytest

from adcssim import __version__
from adcssim.cli import main
from adcssim.telemetry import Telemetry
from tests.conftest import REPO


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "run" in capsys.readouterr().out


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_run_writes_telemetry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            str(REPO / "scenarios" / "case_1a.toml"),
            "--duration",
            "5",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    files = list(tmp_path.glob("case_1a_none_*.parquet"))
    assert len(files) == 1
    assert len(Telemetry.read(files[0])) == 5
    assert "case_1a (none)" in capsys.readouterr().out


def test_run_reports_bad_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(tmp_path / "missing.toml")]) == 2
    assert "error" in capsys.readouterr().err


def test_run_rejects_unknown_backend(capsys: pytest.CaptureFixture[str]) -> None:
    scenario = str(REPO / "scenarios" / "case_1a.toml")
    assert main(["run", scenario, "--backend", "quantum", "--quiet"]) == 2
    assert "unknown ADCS backend" in capsys.readouterr().err
