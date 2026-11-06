"""Golden vectors: recorded ADCS input and output streams that every implementation must
reproduce (Python reference, C, Rust).

Each vector is two files in ``testdata/golden/<variant>/``:

- ``<name>.json``: configuration (the ``adcs_config`` fields), record layout, step count and
  tolerances.
- ``<name>.f64``: little-endian float64 records, one per step: the input fields followed by
  the output fields, in the order given by the JSON ``layout``. Booleans are 0.0/1.0;
  ``mode`` and ``status`` are stored as exact small integers.

Plain binary doubles keep the C and Rust readers to a few lines.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from adcssim.backends.base import AdcsInput, AdcsMode, AdcsOutput
from adcssim.backends.config import AdcsConfig

FORMAT_VERSION = 1

INPUT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("t", 1),
    ("mag_b", 3),
    ("mag_valid", 1),
    ("gyro_b", 3),
    ("gyro_valid", 1),
    ("sun_b", 3),
    ("sun_valid", 1),
    ("sun_ref_i", 3),
    ("mag_ref_i", 3),
)
OUTPUT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("q_est", 4),
    ("w_est", 3),
    ("bias_mag", 3),
    ("bias_gyro", 3),
    ("p_diag", 19),
    ("m_cmd", 3),
    ("mode", 1),
    ("status", 1),
)
INPUT_SIZE = sum(n for _, n in INPUT_LAYOUT)
OUTPUT_SIZE = sum(n for _, n in OUTPUT_LAYOUT)

TOLERANCES = {
    "q_est": 1e-9,
    "w_est": 1e-11,
    "bias_mag": 1e-15,
    "bias_gyro": 1e-11,
    "p_diag": 1e-9,
    "m_cmd": 1e-9,
}
"""Absolute tolerances for comparing another implementation against the reference (double
precision). Mode and status must match exactly. NaN must match NaN."""


@dataclass
class GoldenVector:
    name: str
    description: str
    dt: float
    inertia: NDArray[np.float64]
    max_dipole: NDArray[np.float64]
    config: AdcsConfig
    inputs: list[AdcsInput]
    outputs: list[AdcsOutput]

    # -------------------------------------------------------------------------------------
    def records(self) -> NDArray[np.float64]:
        rows = []
        for inp, out in zip(self.inputs, self.outputs, strict=True):
            row: list[float] = []
            for name, _ in INPUT_LAYOUT:
                row += np.atleast_1d(np.asarray(getattr(inp, name), dtype=np.float64)).tolist()
            for name, _ in OUTPUT_LAYOUT:
                row += np.atleast_1d(np.asarray(getattr(out, name), dtype=np.float64)).tolist()
            rows.append(row)
        return np.asarray(rows, dtype="<f8")

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cfg = {
            f.name: np.asarray(getattr(self.config, f.name)).tolist() for f in fields(self.config)
        }
        meta = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "steps": len(self.inputs),
            "dt": self.dt,
            "inertia": np.asarray(self.inertia).ravel().tolist(),
            "max_dipole": np.asarray(self.max_dipole).tolist(),
            "config": cfg,
            "layout": {
                "input": [[n, w] for n, w in INPUT_LAYOUT],
                "output": [[n, w] for n, w in OUTPUT_LAYOUT],
            },
            "tolerances": TOLERANCES,
        }
        (directory / f"{self.name}.json").write_text(
            json.dumps(meta, indent=1) + "\n", encoding="utf-8"
        )
        self.records().tofile(directory / f"{self.name}.f64")

    @classmethod
    def read(cls, json_path: Path) -> "GoldenVector":
        meta: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"{json_path}: unsupported golden format {meta['format_version']}")
        raw = np.fromfile(json_path.with_suffix(".f64"), dtype="<f8")
        data = raw.reshape(meta["steps"], INPUT_SIZE + OUTPUT_SIZE)
        inputs, outputs = [], []
        for row in data:
            i = 0
            vals: dict[str, Any] = {}
            for name, width in INPUT_LAYOUT:
                vals[name] = row[i] if width == 1 else row[i : i + width].copy()
                i += width
            inputs.append(
                AdcsInput(
                    t=float(vals["t"]),
                    mag_b=vals["mag_b"],
                    mag_valid=bool(vals["mag_valid"]),
                    gyro_b=vals["gyro_b"],
                    gyro_valid=bool(vals["gyro_valid"]),
                    sun_b=vals["sun_b"],
                    sun_valid=bool(vals["sun_valid"]),
                    sun_ref_i=vals["sun_ref_i"],
                    mag_ref_i=vals["mag_ref_i"],
                )
            )
            out: dict[str, Any] = {}
            for name, width in OUTPUT_LAYOUT:
                out[name] = row[i] if width == 1 else row[i : i + width].copy()
                i += width
            outputs.append(
                AdcsOutput(
                    q_est=out["q_est"],
                    w_est=out["w_est"],
                    bias_mag=out["bias_mag"],
                    bias_gyro=out["bias_gyro"],
                    p_diag=out["p_diag"],
                    m_cmd=out["m_cmd"],
                    mode=AdcsMode(int(out["mode"])),
                    status=int(out["status"]),
                )
            )
        cfg = AdcsConfig(**meta["config"])
        return cls(
            name=meta["name"],
            description=meta["description"],
            dt=float(meta["dt"]),
            inertia=np.asarray(meta["inertia"]).reshape(3, 3),
            max_dipole=np.asarray(meta["max_dipole"]),
            config=cfg,
            inputs=inputs,
            outputs=outputs,
        )


def compare(expected: AdcsOutput, actual: AdcsOutput) -> list[str]:
    """Differences beyond tolerance, as human-readable strings (empty if they match)."""
    problems = []
    for name, tol in TOLERANCES.items():
        e = np.asarray(getattr(expected, name), dtype=np.float64)
        a = np.asarray(getattr(actual, name), dtype=np.float64)
        if not np.array_equal(np.isnan(e), np.isnan(a)):
            problems.append(f"{name}: NaN pattern differs ({e} vs {a})")
            continue
        finite = ~np.isnan(e)
        diff = float(np.max(np.abs(e[finite] - a[finite]), initial=0.0))
        if diff > tol:
            problems.append(f"{name}: max difference {diff:.3g} > {tol:g}")
    if int(expected.mode) != int(actual.mode):
        problems.append(f"mode: {int(expected.mode)} vs {int(actual.mode)}")
    if int(expected.status) != int(actual.status):
        problems.append(f"status: {expected.status:#x} vs {actual.status:#x}")
    return problems
