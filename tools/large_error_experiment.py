"""Convergence of the UKF (USQUE) and MEKF filters from large initial attitude errors.

    uv run python tools/large_error_experiment.py

Synthetic constant-rate rotation with realistic sensor noise (sun 5e-3 rad, magnetometer
0.3 µT, gyro 5e-4 rad/s). Each filter starts with an attitude error of the given size and an
honest initial uncertainty of the same size; 20 random cases per size. Counts how many reach
and end below 0.5° within 120 s, and the median time to first reach 0.5°.
"""

import math

import numpy as np

from adcssim.adcs_ref.mekf import Mekf
from adcssim.adcs_ref.usque import Usque, rotation_quat
from adcssim.frames import quat_angle, quat_from_axis_angle, quat_mul, quat_to_dcm

SUN_I = np.array([0.6, 0.8, 0.0])
MAG_I_UT = np.array([10.0, -20.0, 30.0])
NOISE = (5e-4, 3e-6, 0.3, 1e-3, 5e-3)  # gyro, gyro-bias walk, mag, mag-bias walk, sun


def run(filt: Usque | Mekf, error_deg: float, seed: int) -> tuple[float | None, float]:
    rng = np.random.default_rng(seed)
    q = quat_from_axis_angle(rng.normal(size=3), 1.0)
    filt.reset(
        quat_mul(quat_from_axis_angle(rng.normal(size=3), math.radians(error_deg)), q),
        np.r_[np.full(3, math.radians(error_deg) ** 2), np.full(3, 1e-6), np.full(3, 1.0)],
    )
    w = np.radians([2.0, -1.0, 3.0])
    gyro = w.copy()
    first = None
    err = math.inf
    for k in range(1200):
        q = quat_mul(rotation_quat(w, 0.1), q)
        a = quat_to_dcm(q)
        filt.step(
            gyro,
            0.1,
            a @ SUN_I + rng.normal(0, 5e-3, 3),
            SUN_I,
            True,
            a @ MAG_I_UT + rng.normal(0, 0.3, 3),
            MAG_I_UT,
            True,
        )
        gyro = w + rng.normal(0, 5e-4, 3)
        err = math.degrees(float(quat_angle(filt.q_ref, q)))
        if first is None and err < 0.5:
            first = k * 0.1
    return first, err


def main() -> None:
    for error in (10, 30, 60, 90, 150):
        cells = []
        for name in ("UKF", "MEKF"):
            results = [
                run(Usque(1.0, *NOISE) if name == "UKF" else Mekf(*NOISE), error, s)
                for s in range(20)
            ]
            times = [t for t, e in results if t is not None and e < 0.5]
            median = f"{np.median(times):5.1f} s" if times else "    –"
            cells.append(f"{name}: {len(times):2d}/20 converged, median {median}")
        print(f"initial error {error:3d}°   " + "   ".join(cells))


if __name__ == "__main__":
    main()
