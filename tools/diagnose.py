"""Filter diagnostics: consistency of the estimate with its own covariance, innovation
statistics and the accuracy of the FLAE attitude measurement.

    uv run python tools/diagnose.py                      # default scenario set
    uv run python tools/diagnose.py case_2b realistic_1a  # selected scenarios

Writes docs/results/diagnostics.md (tables, regenerated each run) and figures in
docs/results/diagnostics/. The interpretation lives in docs/results/diagnosis.md.

Definitions (per block of the state, using the diagonal of the reported covariance):

- normalised error z = (estimate − truth) / σ. For a consistent filter about 99.7% of |z|
  are below 3 and mean(z²) ≈ 1. mean(z²) ≫ 1 means the filter is overconfident.
- NIS = innovationᵀ P_y⁻¹ innovation per measurement group; its expected value is the group
  size (q: 4, sun: 3, B: 3, ω: 3).
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
from adcssim.frames import quat_angle
from adcssim.runner import Simulation
from adcssim.scenario import load_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "results"
FIG = OUT / "diagnostics"
DEFAULT = ("case_1a", "case_2b", "case_4", "realistic_1a", "detumble")
UT = 1e6
GROUPS = ("q", "sun", "B", "ω")
GROUP_DOF = (4, 3, 3, 3)


def run(name: str) -> tuple[str, Path]:
    sc = load_scenario(REPO / "scenarios" / f"{name}.toml")
    tel = Simulation(sc, create_backend(sc.backend)).run()
    path = REPO / "runs" / f"diag_{name}.parquet"
    tel.write(path)
    return name, path


def _f(tel: Telemetry, name: str) -> np.ndarray:
    return np.asarray(tel.get(name), dtype=np.float64)


def blocks(tel: Telemetry) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(error, σ) per state block, in filter units, shape (N, k)."""
    p = _f(tel, "p_diag")
    q_est, q_true = _f(tel, "q_est"), _f(tel, "q_true")
    sign = np.where(np.sum(q_est * q_true, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    return {
        "q": (q_est - sign * q_true, np.sqrt(p[:, 0:4])),
        "ω (rad/s)": (_f(tel, "w_est") - _f(tel, "w_true"), np.sqrt(p[:, 10:13])),
        "mag bias (µT)": (
            (_f(tel, "bias_mag_est") - _f(tel, "mag_bias_true")) * UT,
            np.sqrt(p[:, 13:16]),
        ),
        "gyro bias (rad/s)": (
            _f(tel, "bias_gyro_est") - _f(tel, "gyro_bias_true"),
            np.sqrt(p[:, 16:19]),
        ),
    }


def analyse(name: str, tel: Telemetry) -> dict[str, Any]:
    t = _f(tel, "t")
    ecl = np.asarray(tel.get("eclipse"), dtype=bool)
    valid = np.isfinite(_f(tel, "attitude_error"))
    warm = valid & (t >= t[np.argmax(valid)] + 60.0)  # skip the first minute after a fix
    res: dict[str, Any] = {"name": name, "consistency": {}, "nis": {}, "flae": {}}
    for block, (err, sig) in blocks(tel).items():
        z = err / sig
        for label, mask in (("sunlit", warm & ~ecl), ("eclipse", warm & ecl)):
            if not mask.any():
                continue
            zz = z[mask]
            res["consistency"][(block, label)] = (
                float(np.mean(np.abs(zz) < 3.0)),
                float(np.mean(zz**2)),
            )
    nis = _f(tel, "nis")
    for g, (group, dof) in enumerate(zip(GROUPS, GROUP_DOF, strict=True)):
        for label, mask in (("sunlit", warm & ~ecl), ("eclipse", warm & ecl)):
            vals = nis[mask, g]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                res["nis"][(group, label)] = (float(np.median(vals)), float(np.mean(vals)), dof)
    q_flae = _f(tel, "q_flae")
    has = np.all(np.isfinite(q_flae), axis=1)
    flae_err = np.full(len(t), np.nan)
    flae_err[has] = np.degrees(quat_angle(q_flae[has], _f(tel, "q_true")[has]))
    est_err = np.degrees(_f(tel, "attitude_error"))
    for label, mask in (("sunlit", warm & ~ecl), ("eclipse", warm & ecl)):
        m = mask & has
        if m.any():
            res["flae"][label] = (
                float(np.median(flae_err[m])),
                float(np.percentile(flae_err[m], 95)),
                float(np.median(est_err[m])),
                float(np.percentile(est_err[m], 95)),
            )
    res["series"] = (t, ecl, est_err, flae_err)
    return res


def figure(res: dict[str, Any], tel: Telemetry, path: Path) -> None:
    t, ecl, est_err, flae_err = res["series"]
    b = blocks(tel)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    # 1. attitude error, FLAE error and the filter's 3σ bound
    ax = axes[0]
    _, q_sig = b["q"]
    bound = np.degrees(2.0 * 3.0 * np.linalg.norm(q_sig[:, :3], axis=1))  # ≈ 3σ angle
    ax.semilogy(t, np.maximum(flae_err, 1e-4), lw=0.4, color="#bbbbbb", label="FLAE measurement")
    ax.semilogy(t, np.maximum(est_err, 1e-4), lw=0.7, color="#1f5fa8", label="filter estimate")
    ax.semilogy(t, bound, lw=0.8, color="#d62728", ls="--", label="filter 3σ (approx.)")
    ax.set_ylabel("attitude error (°)")
    ax.set_ylim(1e-3, 200)
    ax.legend(fontsize=7, loc="upper right")
    # 2. gyro bias error with 3σ
    ax = axes[1]
    err, sig = b["gyro bias (rad/s)"]
    for i, c in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(t, np.degrees(err[:, i]), lw=0.7, color=c, label=f"{'xyz'[i]} error")
        ax.plot(t, np.degrees(3 * sig[:, i]), lw=0.6, color=c, ls=":")
        ax.plot(t, -np.degrees(3 * sig[:, i]), lw=0.6, color=c, ls=":")
    ax.set_ylabel("gyro bias error (°/s)\n(dotted: ±3σ)")
    lim = np.nanpercentile(np.abs(np.degrees(err)), 99) * 1.5 + 1e-3
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    # 3. normalised errors per block (rolling mean of z²)
    ax = axes[2]
    win = max(1, int(60 / (t[1] - t[0])))
    for block, c in zip(b, ("#1f5fa8", "#ff7f0e", "#9467bd", "#2ca02c"), strict=True):
        e, s = b[block]
        z2 = np.nanmean((e / s) ** 2, axis=1)
        roll = np.convolve(np.nan_to_num(z2, nan=0.0), np.ones(win) / win, mode="same")
        ax.semilogy(t, np.maximum(roll, 1e-3), lw=0.8, color=c, label=block)
    ax.axhline(1.0, color="#999", ls="--", lw=0.8)
    ax.set_ylabel("mean z² (60 s)\n1 = consistent")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    # 4. NIS per group
    ax = axes[3]
    nis = _f(tel, "nis")
    for g, (group, c) in enumerate(
        zip(GROUPS, ("#1f5fa8", "#ff7f0e", "#9467bd", "#2ca02c"), strict=True)
    ):
        roll = np.convolve(np.nan_to_num(nis[:, g], nan=0.0), np.ones(win) / win, mode="same")
        ax.semilogy(
            t, np.maximum(roll, 1e-3), lw=0.8, color=c, label=f"{group} (expect {GROUP_DOF[g]})"
        )
    ax.set_ylabel("NIS (60 s mean)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_xlabel("t (s)")
    for a in axes:
        edges = np.flatnonzero(np.diff(np.r_[0, ecl.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2], strict=True):
            a.axvspan(t[s0], t[s1 - 1], color="#000", alpha=0.08, lw=0)
    axes[0].set_title(f"{res['name']}: grey = eclipse", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _n(x: float, d: int = 2) -> str:
    return "–" if not math.isfinite(x) else (f"{x:.{d}f}" if abs(x) < 1e4 else f"{x:.2e}")


def report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Filter diagnostics (generated)",
        "",
        "Generated by `tools/diagnose.py`; the interpretation is in [diagnosis.md](diagnosis.md).",
        "The first 60 s after the first estimate are excluded.",
        "",
        "## Consistency: estimation error against the filter's own σ",
        "",
        "Within 3σ: share of samples with |error| < 3σ (a consistent filter: ~0.997). Mean z²: "
        "mean of (error/σ)² (consistent: ~1; ≫ 1 means overconfident).",
        "",
        "| Scenario | Block | Sunlit: within 3σ / mean z² | Eclipse: within 3σ / mean z² |",
        "|---|---|---|---|",
    ]
    for r in results:
        for block in ("q", "ω (rad/s)", "gyro bias (rad/s)", "mag bias (µT)"):
            cells = []
            for label in ("sunlit", "eclipse"):
                v = r["consistency"].get((block, label))
                cells.append("–" if v is None else f"{v[0]:.3f} / {_n(v[1])}")
            lines.append(f"| {r['name']} | {block} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## Innovations: NIS per measurement group (median / mean; expected mean in brackets)",
        "",
        "| Scenario | Group | Sunlit | Eclipse |",
        "|---|---|---|---|",
    ]
    for r in results:
        for group, dof in zip(GROUPS, GROUP_DOF, strict=True):
            cells = []
            for label in ("sunlit", "eclipse"):
                v = r["nis"].get((group, label))
                cells.append("–" if v is None else f"{_n(v[0])} / {_n(v[1])}")
            lines.append(f"| {r['name']} | {group} ({dof}) | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## FLAE measurement vs filter estimate (attitude error, °: median / p95)",
        "",
        "| Scenario | Sunlit: FLAE | Sunlit: filter | Eclipse: FLAE | Eclipse: filter |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        s, e = r["flae"].get("sunlit"), r["flae"].get("eclipse")
        cells = []
        for v in (s, e):
            if v is None:
                cells += ["–", "–"]
            else:
                cells += [f"{_n(v[0], 3)} / {_n(v[1], 3)}", f"{_n(v[2], 3)} / {_n(v[3], 3)}"]
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    lines += ["", "## Figures", ""]
    for r in results:
        lines += [f"### {r['name']}", "", f"![{r['name']}](diagnostics/{r['name']}.png)", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", default=list(DEFAULT))
    parser.add_argument("--reuse", action="store_true", help="analyse existing runs/diag_*")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        paths = {n: REPO / "runs" / f"diag_{n}.parquet" for n in args.scenarios}
    else:
        with ProcessPoolExecutor(max_workers=len(args.scenarios)) as pool:
            paths = dict(pool.map(run, args.scenarios))
    results = []
    for name in args.scenarios:
        tel = Telemetry.read(paths[name])
        res = analyse(name, tel)
        figure(res, tel, FIG / f"{name}.png")
        results.append(res)
        print(f"analysed {name}")
    (OUT / "diagnostics.md").write_text(report(results), encoding="utf-8")
    print(f"wrote {OUT / 'diagnostics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
