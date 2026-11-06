"""Golden vectors: the reference ADCS must reproduce its own recorded outputs. A failure here
means the algorithm changed; if that was deliberate, regenerate with tools/make_golden.py
(the C and Rust implementations are tested against the same files)."""

from pathlib import Path

import numpy as np
import pytest

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.backends import AdcsContext, AdcsMode, AdcsStatus
from adcssim.golden import INPUT_SIZE, OUTPUT_SIZE, GoldenVector, compare
from tests.conftest import REPO

GOLDEN = sorted((REPO / "testdata" / "golden").glob("*.json"))


def test_golden_set_is_complete() -> None:
    names = {p.stem for p in GOLDEN}
    assert names == {
        "nominal",
        "bad_input",
        "gyro_bias",
        "mag_bias",
        "eclipse_entry",
        "eclipse_start",
        "high_noise",
        "control",
    }
    for p in GOLDEN:
        size = p.with_suffix(".f64").stat().st_size
        assert size % (8 * (INPUT_SIZE + OUTPUT_SIZE)) == 0


@pytest.mark.parametrize("path", GOLDEN, ids=[p.stem for p in GOLDEN])
def test_reference_reproduces_golden_vector(path: Path) -> None:
    g = GoldenVector.read(path)
    adcs = ReferenceAdcs()
    adcs.reset(AdcsContext(dt=g.dt, inertia=g.inertia, max_dipole=g.max_dipole, config=g.config))
    for k, (inp, expected) in enumerate(zip(g.inputs, g.outputs, strict=True)):
        problems = compare(expected, adcs.step(inp))
        assert not problems, f"step {k} (t = {inp.t:.1f} s): {problems}"


def test_golden_vectors_cover_the_intended_paths() -> None:
    def load(name: str) -> GoldenVector:
        return GoldenVector.read(REPO / "testdata" / "golden" / f"{name}.json")

    bad = load("bad_input")
    assert any(o.status & AdcsStatus.BAD_INPUT for o in bad.outputs)
    assert any(o.mode == AdcsMode.SAFE for o in bad.outputs)

    start = load("eclipse_start")
    assert start.outputs[0].mode == AdcsMode.OFF and np.isnan(start.outputs[0].q_est).all()
    assert start.outputs[-1].mode == AdcsMode.ESTIMATE

    entry = load("eclipse_entry")
    assert entry.inputs[0].sun_valid and not entry.inputs[-1].sun_valid

    control = load("control")
    assert any(o.mode == AdcsMode.DETUMBLE for o in control.outputs)
    assert any(np.any(o.m_cmd != 0.0) for o in control.outputs)
    assert any(o.status & AdcsStatus.DIPOLE_SATURATED for o in control.outputs)
