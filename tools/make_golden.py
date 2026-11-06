"""Regenerate the golden vectors in testdata/golden/ from the Python reference ADCS.

    uv run python tools/make_golden.py

Each vector is a short (40 s at 10 Hz) closed-loop run of a thesis scenario with the event
of interest moved near the start, so the files stay small while still covering every code
path: nominal estimation, bias onsets, eclipse entry, starting in eclipse, high noise,
control, and invalid inputs.

Only rerun this after a deliberate change to the reference algorithms; the C and Rust
implementations are tested against these files.
"""

import sys
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.backends import AdcsContext, AdcsInput, AdcsOutput
from adcssim.golden import GoldenVector
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, parse_scenario

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "testdata" / "golden"
DURATION_S = 40.0
ECLIPSE_ENTRY_OFFSET_S = 3816.0 + 1087.0  # TLE epoch → eclipse entry (thesis case 1a timing)
ECLIPSE_EXIT_OFFSET_S = 4913.0 + 1689.3  # case 3a start → sunrise


class Recorder:
    """Wraps a backend and records every input and output."""

    def __init__(self, backend: ReferenceAdcs) -> None:
        self.backend = backend
        self.name = backend.name
        self.inputs: list[AdcsInput] = []
        self.outputs: list[AdcsOutput] = []
        self.ctx: AdcsContext | None = None

    def reset(self, ctx: AdcsContext) -> None:
        self.ctx = ctx
        self.backend.reset(ctx)

    def step(self, inp: AdcsInput) -> AdcsOutput:
        out = self.backend.step(inp)
        self.inputs.append(inp)
        self.outputs.append(out)
        return out


def scenario(name: str, **changes: Any) -> Scenario:
    path = REPO / "scenarios" / f"{name}.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data["time"]["duration_s"] = DURATION_S
    for dotted, value in changes.items():
        node = data
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return parse_scenario(data, base_dir=path.parent)


def record(sc: Scenario) -> tuple[Recorder, AdcsContext]:
    rec = Recorder(ReferenceAdcs())
    Simulation(sc, rec).run()  # type: ignore[arg-type]
    assert rec.ctx is not None
    return rec, rec.ctx


def replay(ctx: AdcsContext, inputs: list[AdcsInput]) -> list[AdcsOutput]:
    adcs = ReferenceAdcs()
    adcs.reset(ctx)
    return [adcs.step(i) for i in inputs]


def corrupt(inputs: list[AdcsInput]) -> list[AdcsInput]:
    """Inject invalid data: NaN gyro at 10 s, Inf magnetometer at 15 s, a missing
    magnetometer at 20–21 s and a zero-length reference field at 25 s."""
    out = []
    for inp in inputs:
        k = round(inp.t * 10)
        if k == 100:
            inp = replace(inp, gyro_b=np.array([np.nan, 0.0, 0.0]))
        elif k == 150:
            inp = replace(inp, mag_b=np.array([np.inf, 0.0, 0.0]))
        elif 200 <= k <= 210:
            inp = replace(inp, mag_valid=False)
        elif k == 250:
            inp = replace(inp, mag_ref_i=np.zeros(3))
        out.append(inp)
    return out


CASES: dict[str, tuple[str, Callable[[], Scenario]]] = {
    "nominal": ("Case 1a, first 40 s.", lambda: scenario("case_1a")),
    "gyro_bias": (
        "Case 2a with the gyro bias [0, 0, 5] °/s active from 10 to 30 s.",
        lambda: scenario(
            "case_2a",
            sensors__gyroscope__bias_windows=[
                {"start_s": 10.0, "end_s": 30.0, "bias_deg_s": [0.0, 0.0, 5.0]}
            ],
        ),
    ),
    "mag_bias": (
        "Case 2b with the magnetometer bias [0, -30, 0] µT active from 10 to 30 s.",
        lambda: scenario(
            "case_2b",
            sensors__magnetometer__bias_windows=[
                {"start_s": 10.0, "end_s": 30.0, "bias_nT": [0.0, -30000.0, 0.0]}
            ],
        ),
    ),
    "eclipse_entry": (
        "Case 1a starting 20 s before eclipse entry: FLAE switches to the filtered sun vector.",
        lambda: scenario("case_1a", time__start_offset_s=ECLIPSE_ENTRY_OFFSET_S - 20.0),
    ),
    "eclipse_start": (
        "Case 3b starting 15 s before sunrise: no fix, then initialisation with a gyro bias.",
        lambda: scenario("case_3b", time__start_offset_s=ECLIPSE_EXIT_OFFSET_S - 15.0),
    ),
    "high_noise": ("Case 4, first 40 s.", lambda: scenario("case_4")),
    "control": (
        "Detumble scenario, first 40 s: B-dot with saturation and duty cycle.",
        lambda: scenario("detumble"),
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (description, make) in CASES.items():
        rec, ctx = record(make())
        # The recorded outputs must be reproducible from the inputs alone.
        again = replay(ctx, rec.inputs)
        for a, b in zip(rec.outputs, again, strict=True):
            assert np.array_equal(a.q_est, b.q_est, equal_nan=True), name
        vectors = [(name, description, rec.inputs, rec.outputs)]
        if name == "nominal":
            bad = corrupt(rec.inputs)
            vectors.append(
                (
                    "bad_input",
                    "Nominal inputs with NaN, Inf, missing and degenerate values injected.",
                    bad,
                    replay(ctx, bad),
                )
            )
        for vname, desc, inputs, outputs in vectors:
            GoldenVector(
                name=vname,
                description=desc,
                dt=ctx.dt,
                inertia=np.asarray(ctx.inertia),
                max_dipole=np.asarray(ctx.max_dipole),
                config=ctx.config,
                inputs=inputs,
                outputs=outputs,
            ).write(OUT)
            errs = [o for o in outputs if np.all(np.isfinite(o.q_est))]
            print(f"{vname:14s} {len(inputs)} steps, {len(errs)} with an estimate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
