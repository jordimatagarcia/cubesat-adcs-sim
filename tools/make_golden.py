"""Generate golden vectors in testdata/golden/<variant>/ from the Python reference ADCS.

    uv run python tools/make_golden.py mekf                  # the MEKF set
    uv run python tools/make_golden.py thesis                # thesis vectors not yet recorded
    uv run python tools/make_golden.py thesis --force        # regenerate the frozen thesis set
    uv run python tools/make_golden.py mekf --only pointing  # selected vectors

Each vector is a short (40 s at 10 Hz) closed-loop run of a thesis scenario with the event
of interest moved near the start, so the files stay small. They cover nominal estimation,
bias onsets, eclipse entry, starting in eclipse, high noise, detumbling, pointing, the
thesis filter's RK4 sub-stepping, and invalid, implausible and degenerate inputs (NaN, Inf,
out-of-range and non-unit values, a gyro outage, FLAE failures). Not covered: covariance
resets, which valid inputs cannot trigger.

Only rerun this after a deliberate change to the reference algorithms; the C and Rust
implementations are tested against these files. The thesis set is frozen (it was recorded
before the detumble scenario moved to 50 Hz dynamics, so regenerating it would change its
control vector): existing thesis vectors are only overwritten with ``--force``, new ones are
added.
"""

import argparse
import dataclasses
import math
import sys
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from adcssim.adcs_ref.adcs import ReferenceAdcs
from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import AdcsContext, AdcsInput, AdcsOutput
from adcssim.golden import GoldenVector
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, parse_scenario

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"
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


def implausible(inputs: list[AdcsInput]) -> list[AdcsInput]:
    """Finite but out-of-range or non-unit values: a 1e165 rad/s gyro reading at 10 s, a sun
    reading of length 1.9 at 12 s, a 1 T magnetometer reading at 14 s, a Sun ephemeris of
    length 0.983 at 16 s, a 5 mT field model at 18 s, and the gyro unavailable 20–26 s."""
    out = []
    for inp in inputs:
        k = round(inp.t * 10)
        if k == 100:
            inp = replace(inp, gyro_b=np.array([1e165, 0.0, 0.0]))
        elif k == 120:
            inp = replace(inp, sun_b=1.9 * np.asarray(inp.sun_b))
        elif k == 140:
            inp = replace(inp, mag_b=np.array([1.0, 0.0, 0.0]))
        elif k == 160:
            inp = replace(inp, sun_ref_i=0.983 * np.asarray(inp.sun_ref_i))
        elif k == 180:
            inp = replace(inp, mag_ref_i=np.array([5e-3, 0.0, 0.0]))
        elif 200 <= k <= 260:
            inp = replace(inp, gyro_valid=False, gyro_b=np.full(3, np.nan))
        out.append(inp)
    return out


def degenerate(inputs: list[AdcsInput]) -> list[AdcsInput]:
    """The sun reading and ephemeris replaced by the field directions (parallel vectors, no
    attitude information): for the first 2 s (FLAE fails at initialisation) and 10–11 s."""
    out = []
    for inp in inputs:
        k = round(inp.t * 10)
        if k <= 20 or 100 <= k <= 110:
            b, r = np.asarray(inp.mag_b), np.asarray(inp.mag_ref_i)
            inp = replace(inp, sun_b=b / np.linalg.norm(b), sun_ref_i=r / np.linalg.norm(r))
        out.append(inp)
    return out


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
    "pointing": (
        "Case 1a at 0.5 °/s with control on, a 5 s settle time and a 30° target about z:"
        " ESTIMATE, then POINT with quaternion feedback.",
        lambda: scenario(
            "case_1a",
            initial__rate_deg_s=[0.3, -0.2, 0.3],
            adcs__control_enabled=True,
            adcs__settle_time=5.0,
            adcs__q_target=[0.0, 0.0, math.sin(math.radians(15)), math.cos(math.radians(15))],
        ),
    ),
    "substeps": (
        "Case 1a with process_dt = dt / 4: the thesis filter's RK4 sub-stepping.",
        lambda: scenario("case_1a", adcs__process_dt=0.025),
    ),
}
DERIVED: dict[str, tuple[str, str, Callable[[list[AdcsInput]], list[AdcsInput]]]] = {
    # name: (source vector, description, input transformation)
    "bad_input": (
        "nominal",
        "Nominal inputs with NaN, Inf, missing and degenerate values injected.",
        lambda x: corrupt(x),
    ),
    "implausible": (
        "nominal",
        "Nominal inputs with finite but implausible or non-unit values and a gyro outage.",
        lambda x: implausible(x),
    ),
    "degenerate": (
        "nominal",
        "Nominal inputs with the sun parallel to the field: FLAE fails.",
        lambda x: degenerate(x),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=("thesis", "mekf"))
    parser.add_argument("--force", action="store_true", help="overwrite frozen thesis vectors")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="write only these vectors")
    args = parser.parse_args()
    out_dir = GOLDEN / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or [*CASES, *DERIVED])
    unknown = wanted - set(CASES) - set(DERIVED)
    if unknown:
        print(f"unknown vectors: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    def frozen(name: str) -> bool:
        return args.variant == "thesis" and (out_dir / f"{name}.json").exists() and not args.force

    for name, (description, make) in CASES.items():
        derived = [d for d, (src, *_) in DERIVED.items() if src == name and d in wanted]
        if name not in wanted and not derived:
            continue
        sc = make()
        if args.variant != "thesis":
            sc = dataclasses.replace(sc, adcs=matched_improved_config(sc, args.variant))
        rec, ctx = record(sc)
        # The recorded outputs must be reproducible from the inputs alone.
        again = replay(ctx, rec.inputs)
        for a, b in zip(rec.outputs, again, strict=True):
            assert np.array_equal(a.q_est, b.q_est, equal_nan=True), name
        vectors = [(name, description, rec.inputs, rec.outputs)] if name in wanted else []
        for d in derived:
            _, desc, transform = DERIVED[d]
            inputs = transform(rec.inputs)
            vectors.append((d, desc, inputs, replay(ctx, inputs)))
        for vname, desc, inputs, outputs in vectors:
            if frozen(vname):
                print(f"{vname:14s} frozen (thesis set): kept; --force to overwrite")
                continue
            GoldenVector(
                name=vname,
                description=desc,
                dt=ctx.dt,
                inertia=np.asarray(ctx.inertia),
                max_dipole=np.asarray(ctx.max_dipole),
                config=ctx.config,
                inputs=inputs,
                outputs=outputs,
            ).write(out_dir)
            errs = [o for o in outputs if np.all(np.isfinite(o.q_est))]
            print(f"{vname:14s} {len(inputs)} steps, {len(errs)} with an estimate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
