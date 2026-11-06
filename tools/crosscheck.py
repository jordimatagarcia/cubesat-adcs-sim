"""Run every scenario with the Python reference and a compiled backend, and report how far
their estimates and commands drift apart over the full run.

    uv run python tools/crosscheck.py            # C library (build adcs-c/build first)
    uv run python tools/crosscheck.py rust

The thesis cases run their own variant (thesis); each scenario also runs with the MEKF,
its noise settings matched to the scenario's sensors.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np

from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import create_backend
from adcssim.runner import Simulation
from adcssim.scenario import load_scenario

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("q_est", "w_est", "bias_gyro_est", "bias_mag_est", "m_cmd", "q_true")


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "c"
    worst = 0.0
    for path in sorted((REPO / "scenarios").glob("*.toml")):
        base = load_scenario(path)
        for variant in (base.adcs.variant, "mekf"):
            sc = base
            if variant != base.adcs.variant:
                sc = dataclasses.replace(base, adcs=matched_improved_config(base, variant))
            if sc.backend == "none":
                sc = dataclasses.replace(sc, backend="python-ref")
            ref = Simulation(sc, create_backend("python-ref")).run()
            other = Simulation(sc, create_backend(backend)).run()
            diffs = {}
            for name in FIELDS:
                a, b = np.asarray(ref.get(name)), np.asarray(other.get(name))
                same_nan = np.array_equal(np.isnan(a), np.isnan(b))
                d = np.abs(a - b)
                diffs[name] = (
                    float(np.nanmax(d))
                    if same_nan and np.any(~np.isnan(d))
                    else (0.0 if same_nan else np.inf)
                )
            modes = np.array_equal(np.asarray(ref.get("mode")), np.asarray(other.get("mode")))
            status = np.array_equal(np.asarray(ref.get("status")), np.asarray(other.get("status")))
            worst = max(worst, *diffs.values())
            cells = "  ".join(f"{k} {v:.1e}" for k, v in diffs.items())
            print(
                f"{path.stem:14s} {variant:7s} {sc.steps:6d} steps  {cells}  "
                f"mode {'=' if modes else '≠'}  status {'=' if status else '≠'}",
                flush=True,
            )
    print(f"largest difference: {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
