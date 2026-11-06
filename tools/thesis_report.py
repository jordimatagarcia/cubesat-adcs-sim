"""Run the thesis test cases with an ADCS backend and write a results report.

    uv run python tools/thesis_report.py                  # all cases, python-ref
    uv run python tools/thesis_report.py --backend c      # (Phase 5) the C library
    uv run python tools/thesis_report.py case_1a case_2b  # selected cases

Writes docs/results/thesis_cases.md (plus figures in docs/results/img/) and keeps each run's
telemetry in runs/ for replay in the viewer.
"""

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adcssim.backends import create_backend
from adcssim.runner import Simulation
from adcssim.scenario import load_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = REPO / "scenarios"
OUT = REPO / "docs" / "results"
CASES = ("case_1a", "case_1b", "case_2a", "case_2b", "case_2c", "case_3a", "case_3b", "case_4")

# What the thesis reports for each case (§5.6), quoted in the report for comparison.
THESIS = {
    "case_1a": "converges within 10 s (gyro ~40 s); < 1° in eclipse; reconverges < 20 s",
    "case_1b": "similar convergence; slightly larger eclipse error; reconverges < 40 s",
    "case_2a": "gyro bias correctly identified, no impact on the state",
    "case_2b": "magnetometer bias not estimated; estimate diverges during the fault",
    "case_2c": "(described in §5.5 but not run in the thesis)",
    "case_3a": "inconclusive in eclipse; ~50 s to converge once valid",
    "case_3b": "gyro bias identified",
    "case_4": "error below 1° at all times, within 0.5° once converged",
}


def run_case(args: tuple[str, str]) -> tuple[str, Path]:
    name, backend = args
    scenario = load_scenario(SCENARIOS / f"{name}.toml")
    telemetry = Simulation(scenario, create_backend(backend)).run()
    path = REPO / "runs" / f"thesis_{name}_{backend}.parquet"
    telemetry.write(path)
    return name, path


def _windows(scenario_name: str) -> list[tuple[float, float, str]]:
    sc = load_scenario(SCENARIOS / f"{scenario_name}.toml")
    sensors = sc.make_sensors()
    out = []
    for label, errors in (
        ("gyro bias", sensors.gyroscope.errors),
        ("mag bias", sensors.magnetometer.errors),
    ):
        out += [(w.start_s, w.end_s, label) for w in errors.bias_windows]
    return out


def _intervals(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(float(t[a]), float(t[b - 1])) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def metrics(name: str, tel: Telemetry) -> dict[str, Any]:
    t = np.asarray(tel.get("t"), dtype=float)
    err = np.degrees(np.asarray(tel.get("attitude_error"), dtype=float))
    ecl = np.asarray(tel.get("eclipse"), dtype=bool)
    dt = float(t[1] - t[0])
    windows = _windows(name)
    in_fault = np.zeros_like(ecl)
    for a, b, _ in windows:
        in_fault |= (t >= a) & (t <= b + 30.0)  # plus a 30 s recovery allowance

    valid = np.isfinite(err)
    first = float(t[np.argmax(valid)]) if valid.any() else math.nan
    # Converged: error below 1° for the next 30 s.
    hold = max(1, round(30.0 / dt))
    below = valid & (err < 1.0)
    run = np.convolve(below.astype(int), np.ones(hold, dtype=int), "valid") == hold
    conv_idx = int(np.argmax(run)) if run.any() else None
    conv_t = float(t[conv_idx]) - first if conv_idx is not None else math.nan

    after = valid & (t >= (t[conv_idx] if conv_idx is not None else math.inf))
    sun_ok = after & ~ecl & ~in_fault
    ecl_ok = after & ecl & ~in_fault

    def stats(mask: np.ndarray) -> tuple[float, float, float]:
        if not mask.any():
            return (math.nan, math.nan, math.nan)
        e = err[mask]
        return float(np.median(e)), float(np.percentile(e, 95)), float(np.max(e))

    out: dict[str, Any] = {
        "first_estimate_s": first,
        "convergence_s": conv_t,
        "sunlit": stats(sun_ok),
        "eclipse": stats(ecl_ok),
        "fault_max": float(np.nanmax(err[in_fault & valid]))
        if (in_fault & valid).any()
        else math.nan,
        "eclipse_windows": _intervals(ecl, t),
        "windows": windows,
    }
    bg_est = np.degrees(np.asarray(tel.get("bias_gyro_est"), dtype=float))
    bg_true = np.degrees(np.asarray(tel.get("gyro_bias_true"), dtype=float))
    bm_est = np.asarray(tel.get("bias_mag_est"), dtype=float) * 1e9
    bm_true = np.asarray(tel.get("mag_bias_true"), dtype=float) * 1e9
    for a, b, label in windows:
        mid = (t >= a + 0.5 * (b - a)) & (t <= b)
        if label == "gyro bias":
            out["gyro_bias_true"] = bg_true[mid].mean(axis=0)
            out["gyro_bias_est"] = np.nanmean(bg_est[mid], axis=0)
        else:
            out["mag_bias_true"] = bm_true[mid].mean(axis=0)
            out["mag_bias_est"] = np.nanmean(bm_est[mid], axis=0)
    if name == "case_3b":  # constant bias: compare at the end of the run
        out["gyro_bias_true"] = bg_true[-1]
        out["gyro_bias_est"] = bg_est[-1]
    return out


def figure(name: str, tel: Telemetry, m: dict[str, Any], path: Path) -> None:
    t = np.asarray(tel.get("t"), dtype=float)
    err = np.degrees(np.asarray(tel.get("attitude_error"), dtype=float))
    has_gyro = any(lbl == "gyro bias" for *_, lbl in m["windows"]) or name == "case_3b"
    has_mag = any(lbl == "mag bias" for *_, lbl in m["windows"])
    rows = 1 + int(has_gyro) + int(has_mag)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 2.6 * rows), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.semilogy(t, np.maximum(err, 1e-3), lw=0.6, color="#1f5fa8")
    ax.axhline(1.0, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("attitude error (°)")
    ax.set_ylim(1e-3, 200)
    for axrow in axes[:, 0]:
        for a, b in m["eclipse_windows"]:
            axrow.axvspan(a, b, color="#000", alpha=0.08, lw=0)
        for a, b, _ in m["windows"]:
            axrow.axvspan(a, b, color="#d62728", alpha=0.12, lw=0)
    ax.set_title(f"{name}: grey = eclipse, red = bias active, dashed = 1°", fontsize=9)
    row = 1
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    if has_gyro:
        ax = axes[row, 0]
        est = np.degrees(np.asarray(tel.get("bias_gyro_est"), dtype=float))
        true = np.degrees(np.asarray(tel.get("gyro_bias_true"), dtype=float))
        for i, c in enumerate(colors):
            ax.plot(t, est[:, i], lw=0.8, color=c, label=f"est {'xyz'[i]}")
            ax.plot(t, true[:, i], lw=1.0, color=c, ls="--")
        ax.set_ylabel("gyro bias (°/s)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        row += 1
    if has_mag:
        ax = axes[row, 0]
        est = np.asarray(tel.get("bias_mag_est"), dtype=float) * 1e6
        true = np.asarray(tel.get("mag_bias_true"), dtype=float) * 1e6
        for i, c in enumerate(colors):
            ax.plot(t, est[:, i], lw=0.8, color=c, label=f"est {'xyz'[i]}")
            ax.plot(t, true[:, i], lw=1.0, color=c, ls="--")
        ax.set_ylabel("mag bias (µT)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1, 0].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x: float, digits: int = 2) -> str:
    return "–" if not math.isfinite(x) else f"{x:.{digits}f}"


def report(results: dict[str, dict[str, Any]], backend: str) -> str:
    lines = [
        "# Thesis test cases: results",
        "",
        f"Backend `{backend}`. Generated by `tools/thesis_report.py`; every number comes from "
        "running the scenario files in `scenarios/` unchanged. How the implementation "
        "differs from the thesis, and why, is in [../adcs.md](../adcs.md).",
        "",
        "Attitude error is the angle between estimated and true attitude. *Converged* means "
        "below 1° for 30 s; statistics are taken after that point and exclude bias windows "
        "(plus 30 s).",
        "",
        "| Case | First estimate (s) | Converged after (s) | Sunlit median / p95 / max (°) "
        "| Eclipse median / p95 / max (°) | Max during bias (°) | Thesis (§5.6) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        s, e = m["sunlit"], m["eclipse"]
        lines.append(
            f"| {name} | {_fmt(m['first_estimate_s'], 1)} | {_fmt(m['convergence_s'], 1)} | "
            f"{_fmt(s[0])} / {_fmt(s[1])} / {_fmt(s[2])} | "
            f"{_fmt(e[0])} / {_fmt(e[1])} / {_fmt(e[2])} | {_fmt(m['fault_max'], 1)} | "
            f"{THESIS[name]} |"
        )
    lines += [
        "",
        "## Bias estimates",
        "",
        "| Case | Bias | True | Estimated |",
        "|---|---|---|---|",
    ]
    for name, m in results.items():
        for key, unit, label in (("gyro", "°/s", "gyroscope"), ("mag", "nT", "magnetometer")):
            if f"{key}_bias_true" in m:
                tr = ", ".join(f"{v:.3g}" for v in m[f"{key}_bias_true"])
                es = ", ".join(f"{v:.3g}" for v in m[f"{key}_bias_est"])
                lines.append(f"| {name} | {label} ({unit}) | [{tr}] | [{es}] |")
    lines += ["", "## Figures", ""]
    for name in results:
        lines += [f"### {name}", "", f"![{name}](img/{name}.png)", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=list(CASES))
    parser.add_argument("--backend", default="python-ref")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    unknown = set(args.cases) - set(CASES)
    if unknown:
        print(f"unknown case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        paths = dict(pool.map(run_case, [(c, args.backend) for c in args.cases]))
    results = {}
    for name in args.cases:
        tel = Telemetry.read(paths[name])
        results[name] = metrics(name, tel)
        figure(name, tel, results[name], OUT / "img" / f"{name}.png")
        print(f"{name}: converged after {results[name]['convergence_s']:.1f} s")
    (OUT / "thesis_cases.md").write_text(report(results, args.backend), encoding="utf-8")
    print(f"wrote {OUT / 'thesis_cases.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
