"""Per-step simulation log, stored as a Parquet table.

One row per ADCS control period. Vectors are split into columns: 3-vectors get the suffixes
``_x, _y, _z`` and quaternions ``_1 … _4`` (``q_true_4`` is the scalar part). The scenario
file, backend and package version are stored in the table metadata, so every log can be
traced back to exactly what produced it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

Column = NDArray[Any]

# name → (number of components, dtype)
SCHEMA: dict[str, tuple[int, type]] = {
    "t": (1, np.float64),
    # Truth
    "r": (3, np.float64),
    "v": (3, np.float64),
    "lat": (1, np.float64),
    "lon": (1, np.float64),
    "alt": (1, np.float64),
    "eclipse": (1, np.bool_),
    "b_i": (3, np.float64),
    "sun_i": (3, np.float64),
    "q_true": (4, np.float64),
    "w_true": (3, np.float64),
    "b_b_true": (3, np.float64),
    "sun_b_true": (3, np.float64),
    "mag_bias_true": (3, np.float64),
    "gyro_bias_true": (3, np.float64),
    # Sensor readings
    "mag": (3, np.float64),
    "mag_valid": (1, np.bool_),
    "gyro": (3, np.float64),
    "gyro_valid": (1, np.bool_),
    "sun": (3, np.float64),
    "sun_valid": (1, np.bool_),
    "sun_faces": (1, np.uint8),
    # ADCS outputs
    "q_est": (4, np.float64),
    "w_est": (3, np.float64),
    "bias_mag_est": (3, np.float64),
    "bias_gyro_est": (3, np.float64),
    "p_diag": (19, np.float64),
    "m_cmd": (3, np.float64),
    "mode": (1, np.uint8),
    "status": (1, np.uint32),
    "compute_time_s": (1, np.float64),
    # Actuation and derived values
    "m_applied": (3, np.float64),
    "tau_control": (3, np.float64),
    "tau_disturbance": (3, np.float64),
    "attitude_error": (1, np.float64),
}

_SUFFIXES = {
    3: ("x", "y", "z"),
    4: ("1", "2", "3", "4"),
    19: tuple(str(i) for i in range(19)),
}


def column_names(name: str) -> list[str]:
    width = SCHEMA[name][0]
    return [name] if width == 1 else [f"{name}_{s}" for s in _SUFFIXES[width]]


@dataclass
class Telemetry:
    """Column arrays of one run, plus metadata."""

    columns: dict[str, Column]
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.columns["t"])

    def get(self, name: str) -> Column:
        """A logged quantity: shape ``(N,)`` for scalars, ``(N, 3)`` or ``(N, 4)`` otherwise."""
        names = column_names(name)
        if len(names) == 1:
            return self.columns[names[0]]
        return np.stack([self.columns[n] for n in names], axis=1)

    def write(self, path: Path) -> None:
        table = pa.table(self.columns)
        meta = {
            **(table.schema.metadata or {}),
            **{k.encode(): v.encode() for k, v in self.metadata.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(meta), path)

    @classmethod
    def read(cls, path: Path) -> "Telemetry":
        table = pq.read_table(path)
        meta = {
            k.decode(): v.decode()
            for k, v in (table.schema.metadata or {}).items()
            if not k.startswith(b"pandas")
        }
        columns = {name: table.column(name).to_numpy() for name in table.column_names}
        return cls(columns, meta)


class TelemetryRecorder:
    """Preallocated column buffers, filled one row at a time."""

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self._n = 0
        self._cols: dict[str, Column] = {}
        for name, (_, dtype) in SCHEMA.items():
            for col in column_names(name):
                self._cols[col] = np.zeros(rows, dtype=dtype)

    def append(self, values: dict[str, object]) -> None:
        if self._n >= self._rows:
            raise IndexError("telemetry buffer is full")
        missing = SCHEMA.keys() - values.keys()
        if missing:
            raise KeyError(f"missing telemetry fields: {sorted(missing)}")
        i = self._n
        for name, value in values.items():
            names = column_names(name)
            if len(names) == 1:
                self._cols[names[0]][i] = np.asarray(value)
            else:
                arr = np.asarray(value)
                for j, col in enumerate(names):
                    self._cols[col][i] = arr[j]
        self._n += 1

    def finish(self, metadata: dict[str, str] | None = None) -> Telemetry:
        cols = {k: v[: self._n].copy() for k, v in self._cols.items()}
        return Telemetry(cols, dict(metadata or {}))
