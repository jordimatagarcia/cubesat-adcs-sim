"""The C library through the simulator's ctypes backend: it must reproduce the golden
vectors and run closed-loop scenarios exactly like the Python reference.

The library is built once per session with CMake (skipped if CMake or a compiler is
missing), so these tests always exercise the current C sources."""

import dataclasses
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import AdcsContext, AdcsStatus, create_backend
from adcssim.backends.native import LibraryNotFoundError, NativeAdcs, library_path
from adcssim.golden import GoldenVector, compare
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, load_scenario
from tests.conftest import REPO

GOLDEN = sorted((REPO / "testdata" / "golden").glob("*/*.json"))


@pytest.fixture(scope="session")
def c_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")
    build = tmp_path_factory.mktemp("adcs-c-build")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO / "adcs-c"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DADCS_BUILD_TESTS=OFF",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True)
    libs = [p for p in build.iterdir() if p.name.startswith(("libadcs_c.", "adcs_c.dll"))]
    so = [p for p in libs if p.suffix in (".so", ".dylib", ".dll")]
    assert so, f"no shared library in {build}"
    return so[0]


@pytest.mark.parametrize("path", GOLDEN, ids=[f"{p.parent.name}/{p.stem}" for p in GOLDEN])
def test_c_library_reproduces_golden_vector(c_library: Path, path: Path) -> None:
    g = GoldenVector.read(path)
    adcs = NativeAdcs("c", c_library)
    adcs.reset(AdcsContext(dt=g.dt, inertia=g.inertia, max_dipole=g.max_dipole, config=g.config))
    for k, (inp, expected) in enumerate(zip(g.inputs, g.outputs, strict=True)):
        problems = compare(expected, adcs.step(inp))
        assert not problems, f"step {k} (t = {inp.t:.1f} s): {problems}"


def _short(name: str, duration_s: float, variant: str | None = None) -> Scenario:
    sc = load_scenario(REPO / "scenarios" / f"{name}.toml")
    sc = dataclasses.replace(sc, duration_s=duration_s)
    if variant is not None:
        sc = dataclasses.replace(sc, adcs=matched_improved_config(sc, variant))
    return sc


@pytest.mark.parametrize(
    ("scenario", "variant"),
    [("detumble", None), ("detumble", "mekf"), ("case_2b", None), ("realistic_1a", "mekf")],
)
def test_closed_loop_matches_reference(c_library: Path, scenario: str, variant: str | None) -> None:
    """Same scenario with the Python and C backends: the commanded dipoles feed back into the
    dynamics, so any difference would grow. They must stay identical to rounding."""
    sc = _short(scenario, 60.0, variant)
    ref = Simulation(sc, create_backend("python-ref")).run()
    c = Simulation(sc, NativeAdcs("c", c_library)).run()
    for name in ("q_true", "w_true", "q_est", "w_est", "m_cmd"):
        a, b = np.asarray(ref.get(name)), np.asarray(c.get(name))
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-8, err_msg=name, equal_nan=True)
    np.testing.assert_array_equal(np.asarray(ref.get("mode")), np.asarray(c.get("mode")))
    np.testing.assert_array_equal(np.asarray(ref.get("status")), np.asarray(c.get("status")))


def test_ukf_variant_is_python_only(c_library: Path) -> None:
    sc = _short("case_1a", 1.0, "improved")
    with pytest.raises(ValueError, match="not implemented"):
        Simulation(sc, NativeAdcs("c", c_library))


def test_step_before_reset_reports_not_initialised(c_library: Path) -> None:
    out = NativeAdcs("c", c_library).step(GoldenVector.read(GOLDEN[0]).inputs[0])
    assert out.status == AdcsStatus.NOT_INITIALISED


def test_missing_library_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ADCSSIM_C_LIB", str(tmp_path / "nope.so"))
    with pytest.raises(LibraryNotFoundError, match="ADCSSIM_C_LIB"):
        library_path("c")
