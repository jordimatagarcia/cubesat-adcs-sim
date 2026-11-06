"""Run every scenario with the Python reference and the compiled backends, and report how
far their estimates and commands drift apart over the full run.

    uv run python tools/crosscheck.py            # C and Rust (build both first)
    uv run python tools/crosscheck.py c          # one backend

Each scenario runs with its own variant (thesis) and with the MEKF, its noise settings
matched to the scenario's sensors. Columns: largest difference of each backend from the
Python reference, and of Rust from C when both run.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np

from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import create_backend
from adcssim.runner import Simulation
from adcssim.scenario import load_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("q_est", "w_est", "bias_gyro_est", "bias_mag_est", "m_cmd", "q_true")


def max_difference(a: Telemetry, b: Telemetry) -> tuple[float, bool]:
    """Largest absolute difference over FIELDS (inf if the NaN patterns differ), and whether
    modes and status flags are identical."""
    worst = 0.0
    for name in FIELDS:
        x, y = np.asarray(a.get(name)), np.asarray(b.get(name))
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return np.inf, False
        d = np.abs(x - y)
        if np.any(~np.isnan(d)):
            worst = max(worst, float(np.nanmax(d)))
    same = all(
        np.array_equal(np.asarray(a.get(k)), np.asarray(b.get(k))) for k in ("mode", "status")
    )
    return worst, same


def main() -> int:
    backends = sys.argv[1:] or ["c", "rust"]
    for path in sorted((REPO / "scenarios").glob("*.toml")):
        base = load_scenario(path)
        for variant in (base.adcs.variant, "mekf"):
            sc = base
            if variant != base.adcs.variant:
                sc = dataclasses.replace(base, adcs=matched_improved_config(base, variant))
            runs = {b: Simulation(sc, create_backend(b)).run() for b in ["python-ref", *backends]}
            cells = []
            for b in backends:
                d, same = max_difference(runs["python-ref"], runs[b])
                cells.append(f"{b} vs python {d:8.1e}{'' if same else ' (modes differ)'}")
            if "c" in runs and "rust" in runs:
                d, same = max_difference(runs["c"], runs["rust"])
                cells.append(f"rust vs c {d:8.1e}{'' if same else ' (modes differ)'}")
            print(
                f"{path.stem:13s} {variant:7s} {sc.steps:6d} steps   " + "   ".join(cells),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
