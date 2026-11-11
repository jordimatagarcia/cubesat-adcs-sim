"""The C and Rust libraries through the simulator's ctypes backend: each must reproduce the
golden vectors and run closed-loop scenarios exactly like the Python reference.

The libraries are built once per session with CMake and Cargo (skipped if the tool is
missing), so these tests always exercise the current sources."""

import ctypes
import dataclasses
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import AdcsConfig, AdcsContext, AdcsStatus, create_backend
from adcssim.backends.abi import config_to_c
from adcssim.backends.native import LibraryNotFoundError, NativeAdcs, library_path
from adcssim.golden import GoldenVector, compare
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, load_scenario
from tests.conftest import REPO
from tests.test_review_fixes import BAD_CONFIGS, INERTIA, MAX_DIPOLE

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


@pytest.fixture(scope="session")
def rust_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cargo") is None:
        pytest.skip("cargo not available")
    target = tmp_path_factory.mktemp("adcs-rs-target")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "adcs-ffi",
            "--manifest-path",
            str(REPO / "adcs-rs" / "Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    release = target / "release"
    so = [p for p in release.iterdir() if p.name.startswith(("libadcs_rs.", "adcs_rs.dll"))]
    so = [p for p in so if p.suffix in (".so", ".dylib", ".dll")]
    assert so, f"no shared library in {release}"
    return so[0]


@pytest.fixture(params=["c", "rust"])
def library(request: pytest.FixtureRequest) -> tuple[str, Path]:
    name: str = request.param
    return name, request.getfixturevalue(f"{name}_library")


@pytest.mark.parametrize("path", GOLDEN, ids=[f"{p.parent.name}/{p.stem}" for p in GOLDEN])
def test_library_reproduces_golden_vector(library: tuple[str, Path], path: Path) -> None:
    g = GoldenVector.read(path)
    adcs = NativeAdcs(*library)
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
def test_closed_loop_matches_reference(
    library: tuple[str, Path], scenario: str, variant: str | None
) -> None:
    """Same scenario with the Python and a compiled backend: the commanded dipoles feed back
    into the dynamics, so any difference would grow. They must stay identical to rounding."""
    sc = _short(scenario, 60.0, variant)
    ref = Simulation(sc, create_backend("python-ref")).run()
    c = Simulation(sc, NativeAdcs(*library)).run()
    for name in ("q_true", "w_true", "q_est", "w_est", "m_cmd"):
        a, b = np.asarray(ref.get(name)), np.asarray(c.get(name))
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-8, err_msg=name, equal_nan=True)
    np.testing.assert_array_equal(np.asarray(ref.get("mode")), np.asarray(c.get("mode")))
    np.testing.assert_array_equal(np.asarray(ref.get("status")), np.asarray(c.get("status")))


def test_ukf_variant_is_python_only(library: tuple[str, Path]) -> None:
    sc = _short("case_1a", 1.0, "improved")
    with pytest.raises(ValueError, match="not implemented"):
        Simulation(sc, NativeAdcs(*library))


def test_step_before_reset_reports_not_initialised(library: tuple[str, Path]) -> None:
    out = NativeAdcs(*library).step(GoldenVector.read(GOLDEN[0]).inputs[0])
    assert out.status == AdcsStatus.NOT_INITIALISED


def test_missing_library_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ADCSSIM_C_LIB", str(tmp_path / "nope.so"))
    with pytest.raises(LibraryNotFoundError, match="ADCSSIM_C_LIB"):
        library_path("c")


@pytest.mark.parametrize("case", BAD_CONFIGS, ids=[r for r, *_ in BAD_CONFIGS])
def test_libraries_reject_what_the_reference_rejects(
    library: tuple[str, Path], case: tuple[str, dict[str, object], dict[str, object]]
) -> None:
    """One configuration rule set: every configuration the Python reference rejects,
    adcs_init rejects too. The bad value is set on the C struct directly, so this also
    covers values Python already refuses when building the config object."""
    _, cfg_changes, ctx_changes = case
    adcs = NativeAdcs(*library)
    base = AdcsConfig(variant=str(cfg_changes.get("variant", "thesis")))
    c_cfg = config_to_c(base, 0.1, INERTIA, MAX_DIPOLE)
    state = ctypes.addressof(adcs._state)
    assert adcs.lib.adcs_init(state, ctypes.byref(c_cfg)) == 0  # the valid default
    for name, value in (cfg_changes | ctx_changes).items():
        if name == "variant":
            continue
        field = getattr(c_cfg, name)
        if isinstance(field, ctypes.Array):
            data = np.asarray(value, dtype=np.float64).ravel()
            setattr(c_cfg, name, type(field)(*data))
        else:
            setattr(c_cfg, name, value)
    assert adcs.lib.adcs_init(state, ctypes.byref(c_cfg)) == -3
