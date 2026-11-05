"""Fixed-capacity history of telemetry rows for plotting, without per-frame reallocations."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from adcssim.telemetry import SCHEMA

Row = dict[str, Any]

FIELDS: dict[str, int] = {name: width for name, (width, _) in SCHEMA.items()} | {"gmst": 1}


class History:
    """The most recent ``capacity`` rows, per field.

    Data is kept in buffers of twice the capacity; when they fill up, the newest half is moved
    to the front. Reading the last ``k`` rows is then always a contiguous slice.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._n = 0
        self._buf: dict[str, NDArray[np.float64]] = {
            name: np.full((2 * capacity, width) if width > 1 else 2 * capacity, np.nan)
            for name, width in FIELDS.items()
        }

    def __len__(self) -> int:
        return min(self._n, self.capacity)

    def clear(self) -> None:
        self._n = 0

    def append(self, row: Row) -> None:
        if self._n == 2 * self.capacity:
            for buf in self._buf.values():
                buf[: self.capacity] = buf[self.capacity :]
            self._n = self.capacity
        for name, buf in self._buf.items():
            buf[self._n] = row[name]
        self._n += 1

    def extend(self, rows: list[Row]) -> None:
        for row in rows:
            self.append(row)

    def last(self, name: str, k: int | None = None) -> NDArray[np.float64]:
        """The last ``k`` values of a field (all stored values if ``k`` is None)."""
        k = len(self) if k is None else min(k, len(self))
        return self._buf[name][self._n - k : self._n]

    def latest(self, name: str) -> NDArray[np.float64]:
        if self._n == 0:
            raise IndexError("history is empty")
        return self._buf[name][self._n - 1]
