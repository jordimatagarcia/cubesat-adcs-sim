"""Run every scenario with the thesis, improved (UKF) and MEKF variants and compare them.

    uv run python tools/compare_variants.py            # all scenarios
    uv run python tools/compare_variants.py case_2b    # selected scenarios

The thesis variant runs each scenario file unchanged; the improved and MEKF variants run the
same scenario with their noise settings matched to the scenario's sensors
(``adcssim.adcs_ref.tuning``). Writes docs/results/variants.md and figures in
docs/results/variants/.
"""

import argparse
import dataclasses
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adcssim.adcs_ref.tuning import matched_improved_config
from adcssim.backends import create_backend
from adcssim.runner import Simulation
from adcssim.scenario import load_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "results"
FIG = OUT / "variants"
SCENARIOS = (
    "case_1a",
    "case_1b",
    "case_2a",
    "case_2b",
    "case_2c",
    "case_3a",
    "case_3b",
    "case_4",
    "realistic_1a",
    "detumble",
)
VARIANTS = ("thesis", "improved", "mekf")


def run(job: tuple[str, str]) -> tuple[str, str, Path]:
    name, variant = job
    sc = load_scenario(REPO / "scenarios" / f"{name}.toml")
    if variant != "thesis":
        sc = dataclasses.replace(sc, adcs=matched_improved_config(sc, variant))
    tel = Simulation(sc, create_backend("python-ref")).run()
    path = REPO / "runs" / f"variant_{name}_{variant}.parquet"
    tel.write(path)
    return name, variant, path


def fault_windows(name: str) -> list[tuple[float, float]]:
    sensors = load_scenario(REPO / "scenarios" / f"{name}.toml").make_sensors()
    return [
        (w.start_s, w.end_s)
        for errors in (sensors.gyroscope.errors, sensors.magnetometer.errors)
        for w in errors.bias_windows
    ]


def attitude_sigma(tel: Telemetry) -> np.ndarray:
    """1σ of the attitude error angle (rad) from the reported covariance, either variant."""
    p = np.asarray(tel.get("p_diag"), dtype=float)
    improved = np.isnan(p[:, 3])
    thesis_var = 4.0 * p[:, 0:3].sum(axis=1)  # small-angle: θ ≈ 2|δq_v|
    improved_var = p[:, 0:3].sum(axis=1)
    return np.sqrt(np.where(improved, improved_var, thesis_var))


def metrics(name: str, tel: Telemetry) -> dict[str, Any]:
    t = np.asarray(tel.get("t"), dtype=float)
    err = np.degrees(np.asarray(tel.get("attitude_error"), dtype=float))
    ecl = np.asarray(tel.get("eclipse"), dtype=bool)
    valid = np.isfinite(err)
    warm = valid & (t >= (t[np.argmax(valid)] if valid.any() else 0.0) + 60.0)
    fault = np.zeros_like(ecl)
    for a, b in fault_windows(name):
        fault |= (t >= a) & (t <= b + 30.0)

    def stats(mask: np.ndarray) -> tuple[float, float, float]:
        if not mask.any():
            return (math.nan,) * 3
        e = err[mask]
        return float(np.median(e)), float(np.percentile(e, 95)), float(np.max(e))

    sigma = np.degrees(attitude_sigma(tel))
    inside = (err <= 3.0 * sigma)[warm]
    bg = np.degrees(np.asarray(tel.get("bias_gyro_est") - tel.get("gyro_bias_true"), dtype=float))
    steady = warm & ~fault
    return {
        "first_estimate": float(t[np.argmax(valid)]) if valid.any() else math.nan,
        "sunlit": stats(warm & ~ecl & ~fault),
        "eclipse": stats(warm & ecl & ~fault),
        "fault_max": stats(valid & fault)[2],
        "consistency": float(np.mean(inside)) if inside.size else math.nan,
        "gyro_bias_rms": float(np.sqrt(np.nanmean(bg[steady] ** 2))) if steady.any() else math.nan,
        "compute_ms": float(np.mean(tel.get("compute_time_s")) * 1e3),
        "series": (t, ecl, err),
    }


def figure(name: str, results: dict[str, dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colours = {"thesis": "#d62728", "improved": "#1f5fa8", "mekf": "#2ca02c"}
    for variant, m in results.items():
        t, ecl, err = m["series"]
        ax.semilogy(t, np.maximum(err, 1e-3), lw=0.6, color=colours[variant], label=variant)
    t, ecl, _ = next(iter(results.values()))["series"]
    edges = np.flatnonzero(np.diff(np.r_[0, ecl.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        ax.axvspan(t[a], t[b - 1], color="#000", alpha=0.08, lw=0)
    for a, b in fault_windows(name):
        ax.axvspan(a, b, color="#d62728", alpha=0.1, lw=0)
    ax.axhline(1.0, color="#999", ls="--", lw=0.8)
    ax.set_ylim(1e-3, 200)
    ax.set_ylabel("attitude error (°)")
    ax.set_xlabel("t (s)")
    ax.set_title(f"{name}: grey = eclipse, red = bias fault, dashed = 1°", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _n(x: float, d: int = 2) -> str:
    return "–" if not math.isfinite(x) else f"{x:.{d}f}"


def _trio(v: tuple[float, float, float]) -> str:
    return " / ".join(_n(x) for x in v)


def report(all_results: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "# ADCS variants compared: thesis, improved (UKF), MEKF (generated)",
        "",
        "Generated by `tools/compare_variants.py`. Same scenarios and noise realisations for "
        "all variants; the improved and MEKF variants' noise settings are matched to each "
        "scenario's sensors. Errors are attitude errors in degrees (median / p95 / max), "
        "excluding the first 60 s and, except in the fault column, the bias-fault windows "
        "plus 30 s. "
        "*Within 3σ*: share of samples where the error is inside the filter's own 3σ bound "
        "(a consistent filter: ~0.99). What the variants are: [../adcs.md](../adcs.md).",
        "",
        "| Scenario | Variant | Sunlit | Eclipse | Max in fault | Within 3σ "
        "| Gyro-bias RMS error (°/s) | ms/step |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, results in all_results.items():
        for variant, m in results.items():
            lines.append(
                f"| {name} | {variant} | {_trio(m['sunlit'])} | {_trio(m['eclipse'])} | "
                f"{_n(m['fault_max'])} | {_n(m['consistency'], 3)} | "
                f"{_n(m['gyro_bias_rms'], 4)} | {_n(m['compute_ms'])} |"
            )
    lines += ["", "## Figures", ""]
    for name in all_results:
        lines += [f"### {name}", "", f"![{name}](variants/{name}.png)", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--reuse", action="store_true", help="analyse existing runs/variant_*")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [(n, v) for n in args.scenarios for v in VARIANTS]
    if args.reuse:
        done = [(n, v, REPO / "runs" / f"variant_{n}_{v}.parquet") for n, v in jobs]
    else:
        # Longest runs first so the pool stays busy.
        jobs.sort(key=lambda j: -load_scenario(REPO / "scenarios" / f"{j[0]}.toml").duration_s)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            done = list(pool.map(run, jobs))
    paths = {(n, v): p for n, v, p in done}
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for name in args.scenarios:
        all_results[name] = {v: metrics(name, Telemetry.read(paths[(name, v)])) for v in VARIANTS}
        figure(name, all_results[name], FIG / f"{name}.png")
        print(f"{name}: done")
    (OUT / "variants.md").write_text(report(all_results), encoding="utf-8")
    print(f"wrote {OUT / 'variants.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
