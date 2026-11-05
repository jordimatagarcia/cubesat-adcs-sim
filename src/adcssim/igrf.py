"""International Geomagnetic Reference Field, 14th generation (IGRF-14).

Spherical-harmonic synthesis of the main field from the official IAGA coefficient file
(``data/igrf14coeffs.txt``, Schmidt semi-normalised, nT). The Legendre recursion and the
Schmidt factors follow R. B. Davis (2004), *Mathematical Modeling of Earth's Magnetic Field*,
Virginia Tech technical note, and Wertz (1978) appendix H.

Valid from 1900.0 to 2030.0. Between model epochs the coefficients are interpolated linearly;
after the last epoch the published secular variation is extrapolated.
"""

import math
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adcssim.frames import ecef_to_spherical, spherical_to_ecef_vector

FloatArray = NDArray[np.float64]

IGRF_REFERENCE_RADIUS = 6_371_200.0
"""Reference radius ``a`` of the IGRF model (m)."""

_SV_YEARS = 5.0


class Igrf:
    """IGRF main-field model loaded from an IAGA coefficient file."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            text = (
                resources.files("adcssim")
                .joinpath("data/igrf14coeffs.txt")
                .read_text(encoding="utf-8")
            )
        else:
            text = path.read_text(encoding="utf-8")
        self.epochs, self._g, self._h, self._g_sv, self._h_sv = _parse(text)
        self.nmax = self._g.shape[1] - 1
        self._schmidt = _schmidt_factors(self.nmax)

    @property
    def valid_range(self) -> tuple[float, float]:
        return float(self.epochs[0]), float(self.epochs[-1] + _SV_YEARS)

    def coefficients(self, year: float) -> tuple[FloatArray, FloatArray]:
        """Schmidt semi-normalised ``g[n, m]`` and ``h[n, m]`` (nT) at a decimal year."""
        lo, hi = self.valid_range
        if not lo <= year <= hi:
            raise ValueError(f"IGRF is valid for {lo:.1f}–{hi:.1f}, got {year:.3f}")
        last = len(self.epochs) - 1
        if year >= self.epochs[last]:
            dt = year - self.epochs[last]
            return self._g[last] + self._g_sv * dt, self._h[last] + self._h_sv * dt
        i = int(np.searchsorted(self.epochs, year, side="right")) - 1
        w = (year - self.epochs[i]) / (self.epochs[i + 1] - self.epochs[i])
        g = (1.0 - w) * self._g[i] + w * self._g[i + 1]
        h = (1.0 - w) * self._h[i] + w * self._h[i + 1]
        return g, h

    def field_spherical(
        self, r: ArrayLike, colat: ArrayLike, lon: ArrayLike, year: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Field components ``(B_r, B_θ, B_φ)`` in nT at geocentric radius ``r`` (m),
        colatitude and longitude (rad). Inputs may be arrays of equal shape."""
        rr = np.asarray(r, dtype=np.float64)
        th = np.asarray(colat, dtype=np.float64)
        ph = np.asarray(lon, dtype=np.float64)
        g, h = self.coefficients(year)
        # Convert Schmidt semi-normalised coefficients to Gauss-normalised ones.
        g = g * self._schmidt
        h = h * self._schmidt

        nmax = self.nmax
        ct, st = np.cos(th), np.sin(th)
        p = np.zeros((nmax + 1, nmax + 1, *th.shape))
        dp = np.zeros_like(p)
        p[0, 0] = 1.0
        for n in range(1, nmax + 1):
            for m in range(n + 1):
                if m == n:
                    p[n, n] = st * p[n - 1, n - 1]
                    dp[n, n] = st * dp[n - 1, n - 1] + ct * p[n - 1, n - 1]
                elif n == 1:
                    p[1, 0] = ct * p[0, 0]
                    dp[1, 0] = ct * dp[0, 0] - st * p[0, 0]
                else:
                    k = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    p[n, m] = ct * p[n - 1, m] - k * p[n - 2, m]
                    dp[n, m] = ct * dp[n - 1, m] - st * p[n - 1, m] - k * dp[n - 2, m]

        ar = IGRF_REFERENCE_RADIUS / rr
        b_r = np.zeros_like(th)
        b_t = np.zeros_like(th)
        b_p = np.zeros_like(th)
        for m in range(nmax + 1):
            cm, sm = np.cos(m * ph), np.sin(m * ph)
            for n in range(max(m, 1), nmax + 1):
                arn = ar ** (n + 2)
                gc_hs = g[n, m] * cm + h[n, m] * sm
                b_r += (n + 1) * arn * gc_hs * p[n, m]
                b_t -= arn * gc_hs * dp[n, m]
                b_p += m * arn * (g[n, m] * sm - h[n, m] * cm) * p[n, m]
        # B_φ carries 1/sin θ; every m ≥ 1 term of P has a sin θ factor, so this is finite at
        # the poles. Guard the division numerically.
        b_p = b_p / np.where(np.abs(st) < 1e-12, 1e-12, st)
        return b_r, b_t, b_p

    def field_ecef(self, r_ecef: ArrayLike, year: float) -> FloatArray:
        """Field vector(s) in Earth-fixed coordinates, in tesla."""
        rad, colat, lon = ecef_to_spherical(r_ecef)
        b_r, b_t, b_p = self.field_spherical(rad, colat, lon, year)
        return spherical_to_ecef_vector(b_r, b_t, b_p, colat, lon) * 1e-9


def _schmidt_factors(nmax: int) -> FloatArray:
    """Factors ``S[n, m]`` that turn Schmidt semi-normalised into Gauss-normalised values."""
    s = np.zeros((nmax + 1, nmax + 1))
    s[0, 0] = 1.0
    for n in range(1, nmax + 1):
        s[n, 0] = s[n - 1, 0] * (2 * n - 1) / n
        for m in range(1, n + 1):
            factor = 2.0 if m == 1 else 1.0
            s[n, m] = s[n, m - 1] * math.sqrt((n - m + 1) * factor / (n + m))
    return s


def _parse(
    text: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    epochs: list[float] | None = None
    rows: list[tuple[str, int, int, list[float]]] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or line.startswith("#"):
            continue
        if fields[0] == "g/h":
            # Header: "g/h n m 1900.0 1905.0 ... 2025.0 2025-30"
            epochs = [float(x) for x in fields[3:-1]]
        elif fields[0] in ("g", "h"):
            values = [float(x) for x in fields[3:]]
            rows.append((fields[0], int(fields[1]), int(fields[2]), values))
    if epochs is None or not rows:
        raise ValueError("not an IGRF coefficient file")

    nmax = max(n for _, n, _, _ in rows)
    k = len(epochs)
    g = np.zeros((k, nmax + 1, nmax + 1))
    h = np.zeros_like(g)
    g_sv = np.zeros((nmax + 1, nmax + 1))
    h_sv = np.zeros_like(g_sv)
    for kind, n, m, values in rows:
        if len(values) != k + 1:
            raise ValueError(f"IGRF row {kind} {n} {m}: expected {k + 1} values")
        target, target_sv = (g, g_sv) if kind == "g" else (h, h_sv)
        target[:, n, m] = values[:k]
        target_sv[n, m] = values[k]
    return np.array(epochs), g, h, g_sv, h_sv
