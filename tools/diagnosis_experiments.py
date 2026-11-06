"""Hypothesis experiments behind docs/results/diagnosis.md.

    uv run python tools/diagnosis_experiments.py eclipse    # H1: q pseudo-measurement in eclipse
    uv run python tools/diagnosis_experiments.py duty       # H2: coil duty cycle
    uv run python tools/diagnosis_experiments.py v4         # H3: measurement noise in case 4
    uv run python tools/diagnosis_experiments.py magbias    # H4: magnetometer-bias tuning
    uv run python tools/diagnosis_experiments.py detumble   # control vs sun-sensor coverage

Each experiment runs a scenario with one thing changed and prints the attitude error
(median / p95 / max, °) in sunlight and in eclipse. H1 disables the eclipse attitude
measurement by patching the reference ADCS at runtime; nothing here changes the library.
"""

import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

import adcssim.adcs_ref.adcs as reference
from adcssim.runner import Simulation
from adcssim.scenario import parse_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parent.parent


def scenario_data(name: str, duration: float | None = None, **adcs: Any) -> dict[str, Any]:
    data: dict[str, Any] = tomllib.loads((REPO / "scenarios" / f"{name}.toml").read_text())
    if duration is not None:
        data["time"]["duration_s"] = duration
    data.setdefault("adcs", {}).update(adcs)
    return data


def run(data: dict[str, Any]) -> Telemetry:
    return Simulation(parse_scenario(data, REPO / "scenarios")).run()


def summary(label: str, tel: Telemetry, extra: str = "") -> None:
    t = tel.get("t")
    err = np.degrees(tel.get("attitude_error"))
    ecl = tel.get("eclipse").astype(bool)
    ok = np.isfinite(err) & (t > 60)

    def stats(a: np.ndarray) -> str:
        if not len(a):
            return "–"
        return f"{np.median(a):6.3f} / {np.percentile(a, 95):7.3f} / {np.max(a):7.2f}"

    print(f"{label:40s} sunlit {stats(err[ok & ~ecl])}   eclipse {stats(err[ok & ecl])}  {extra}")


def eclipse() -> None:
    for name in ("case_1a", "realistic_1a"):
        data = scenario_data(name, 3000)
        summary(f"{name}: FLAE with filtered sun (as is)", run(data))
        original = reference.propagate_body_vector
        reference.propagate_body_vector = lambda *a: None  # type: ignore[assignment]
        try:
            summary(f"{name}: no q measurement in eclipse", run(data))
        finally:
            reference.propagate_body_vector = original


def duty() -> None:
    for duty_cycle in (0.8, 1.0):
        data = scenario_data("detumble", 3000)
        data["actuators"]["duty_cycle"] = duty_cycle
        tel = run(data)
        t = tel.get("t")
        bias_err = np.degrees(np.abs(tel.get("bias_gyro_est") - tel.get("gyro_bias_true")))
        worst = np.nanmax(bias_err.max(axis=1)[t > 60])
        summary(f"detumble, duty cycle {duty_cycle}", tel, f"max gyro-bias error {worst:.3f} °/s")


def v4() -> None:
    summary("case_4, thesis V", run(scenario_data("case_4")))
    matched = scenario_data("case_4", v=[2.5e-3, 1e-2, 1e-6, 2.5e-5])
    summary("case_4, V matched to the sensors", run(matched))


def magbias() -> None:
    tel = run(scenario_data("case_2b", 400, p_b0=[100.0, 1e-7], w_b=[1e-4, 1e-6]))
    t = tel.get("t")
    window = (t > 150) & (t < 225)
    true = tel.get("mag_bias_true")[window].mean(axis=0) * 1e6
    est = tel.get("bias_mag_est")[window].mean(axis=0) * 1e6
    err = np.degrees(tel.get("attitude_error"))
    print(
        f"case_2b, loose mag-bias tuning: true {np.round(true, 1)} µT, "
        f"estimated {np.round(est, 2)} µT, max attitude error during the fault "
        f"{np.nanmax(err[(t > 125) & (t < 255)]):.1f}°"
    )


def detumble() -> None:
    variants: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("control on, 45° sun FOV (as is)", {}, {}),
        ("control off", {"control_enabled": False}, {}),
        ("full-sphere sun coverage", {}, {"half_fov_deg": 90.0}),
        ("control off + full coverage", {"control_enabled": False}, {"half_fov_deg": 90.0}),
    ]
    for label, adcs, sun in variants:
        data = scenario_data("detumble", 3000, **adcs)
        data["sensors"]["sun"].update(sun)
        tel = run(data)
        ecl = tel.get("eclipse").astype(bool)
        rate = np.degrees(np.linalg.norm(tel.get("w_true"), axis=1))
        summary(label, tel, f"| body rate in eclipse {np.median(rate[ecl]):.1f} °/s")


EXPERIMENTS = {"eclipse": eclipse, "duty": duty, "v4": v4, "magbias": magbias, "detumble": detumble}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        sys.exit(f"usage: diagnosis_experiments.py {{{','.join(EXPERIMENTS)}}}")
    EXPERIMENTS[sys.argv[1]]()
