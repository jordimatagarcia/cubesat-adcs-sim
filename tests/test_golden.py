"""Golden vectors: the reference ADCS must reproduce its own recorded outputs. A failure here
means the algorithm changed; if that was deliberate, regenerate with tools/make_golden.py
(the C and Rust implementations are tested against the same files)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.backends import AdcsContext, AdcsMode, AdcsStatus
from adcssim.golden import INPUT_SIZE, OUTPUT_SIZE, GoldenVector, compare
from tests.conftest import REPO

GOLDEN = sorted((REPO / "testdata" / "golden").glob("*/*.json"))
NAMES = {
    "nominal",
    "bad_input",
    "gyro_bias",
    "mag_bias",
    "eclipse_entry",
    "eclipse_start",
    "high_noise",
    "control",
    "pointing",
    "substeps",
    "implausible",
    "degenerate",
}


@pytest.mark.parametrize("variant", ["thesis", "mekf"])
def test_golden_set_is_complete(variant: str) -> None:
    files = [p for p in GOLDEN if p.parent.name == variant]
    names = {p.stem for p in files}
    assert names == NAMES
    for p in files:
        assert GoldenVector.read(p).config.variant == variant
    for p in files:
        size = p.with_suffix(".f64").stat().st_size
        assert size % (8 * (INPUT_SIZE + OUTPUT_SIZE)) == 0


@pytest.mark.parametrize("path", GOLDEN, ids=[f"{p.parent.name}/{p.stem}" for p in GOLDEN])
def test_reference_reproduces_golden_vector(path: Path) -> None:
    g = GoldenVector.read(path)
    adcs = ReferenceAdcs()
    adcs.reset(AdcsContext(dt=g.dt, inertia=g.inertia, max_dipole=g.max_dipole, config=g.config))
    for k, (inp, expected) in enumerate(zip(g.inputs, g.outputs, strict=True)):
        problems = compare(expected, adcs.step(inp))
        assert not problems, f"step {k} (t = {inp.t:.1f} s): {problems}"


@pytest.mark.parametrize("variant", ["thesis", "mekf"])
def test_golden_vectors_cover_the_intended_paths(variant: str) -> None:
    def load(name: str) -> GoldenVector:
        return GoldenVector.read(REPO / "testdata" / "golden" / variant / f"{name}.json")

    bad = load("bad_input")
    assert any(o.status & AdcsStatus.BAD_INPUT for o in bad.outputs)
    assert any(o.mode == AdcsMode.SAFE for o in bad.outputs)

    start = load("eclipse_start")
    assert start.outputs[0].mode == AdcsMode.OFF and np.isnan(start.outputs[0].q_est).all()
    assert start.outputs[-1].mode == AdcsMode.ESTIMATE

    entry = load("eclipse_entry")
    assert entry.inputs[0].sun_valid and not entry.inputs[-1].sun_valid

    pointing = load("pointing")
    assert any(o.mode == AdcsMode.POINT for o in pointing.outputs)
    assert any(o.status & AdcsStatus.FLAE_FAILED for o in load("degenerate").outputs)
    implausible = load("implausible")
    assert any(o.status & AdcsStatus.BAD_INPUT for o in implausible.outputs)
    if variant == "mekf":
        assert any(o.status & AdcsStatus.RATE_PROPAGATED for o in implausible.outputs)
    else:
        assert load("substeps").config.process_dt < load("substeps").dt

    control = load("control")
    assert any(o.mode == AdcsMode.DETUMBLE for o in control.outputs)
    assert any(np.any(o.m_cmd != 0.0) for o in control.outputs)
    assert any(o.status & AdcsStatus.DIPOLE_SATURATED for o in control.outputs)


def test_mekf_vectors_record_the_fault_detector() -> None:
    def load(name: str) -> GoldenVector:
        return GoldenVector.read(REPO / "testdata" / "golden" / "mekf" / f"{name}.json")

    for name in ("gyro_bias", "mag_bias"):
        assert any(o.status & AdcsStatus.BIAS_FAULT for o in load(name).outputs), name
    assert not any(o.status & AdcsStatus.BIAS_FAULT for o in load("nominal").outputs)


def test_compare_catches_differences_hidden_by_infinities_and_small_values() -> None:
    """Was: equal infinities gave a NaN difference that hid every other difference in the
    field, and the magnetometer-bias tolerance (1e-13 T) exceeded the recorded values."""
    g = GoldenVector.read(GOLDEN[0])
    out = g.outputs[-1]
    p = out.p_diag.copy()
    p[0], p[1] = np.inf, p[1] + 0.5
    hidden = dataclasses.replace(out, p_diag=p)
    assert compare(
        dataclasses.replace(out, p_diag=np.where(np.arange(19) == 0, np.inf, out.p_diag)), hidden
    )
    small = dataclasses.replace(out, bias_mag=np.array([5e-14, 0.0, 0.0]))
    assert compare(small, dataclasses.replace(small, bias_mag=np.zeros(3)))
