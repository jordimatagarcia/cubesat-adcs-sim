import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from adcssim.frames import ecef_to_spherical, geodetic_to_ecef
from adcssim.igrf import Igrf
from adcssim.timebase import decimal_year


@pytest.fixture(scope="module")
def igrf() -> Igrf:
    return Igrf()


def test_model_metadata(igrf: Igrf) -> None:
    assert igrf.nmax == 13
    assert igrf.valid_range == (1900.0, 2030.0)


def test_main_dipole_coefficient(igrf: Igrf) -> None:
    g, _ = igrf.coefficients(2020.0)
    assert g[1, 0] == -29403.41  # IGRF-14, epoch 2020.0


def test_matches_independent_implementation(igrf: Igrf, igrf_ref: list[dict[str, Any]]) -> None:
    for point in igrf_ref:
        t = datetime.fromisoformat(point["date"]).replace(tzinfo=UTC)
        b = igrf.field_spherical(
            point["r_km"] * 1e3,
            math.radians(point["theta"]),
            math.radians(point["phi"]),
            decimal_year(t),
        )
        # Exact at model epochs; elsewhere the two libraries count decimal years slightly
        # differently (a few hundredths of a nT out of ~40 000 nT).
        tol = 1e-6 if t.month == 1 and t.day == 1 and t.hour == 0 else 0.2
        np.testing.assert_allclose([float(x) for x in b], point["B"], atol=tol, err_msg=str(point))


def test_rejects_dates_outside_model(igrf: Igrf) -> None:
    with pytest.raises(ValueError, match="valid"):
        igrf.coefficients(2031.0)


def test_field_magnitude_in_leo(igrf: Igrf) -> None:
    lats = np.radians(np.linspace(-89, 89, 37))
    lons = np.radians(np.linspace(-180, 180, 37))
    lat, lon = np.meshgrid(lats, lons)
    r = geodetic_to_ecef(lat, lon, np.full_like(lat, 500e3))
    magnitude = np.linalg.norm(igrf.field_ecef(r, 2021.8), axis=-1)
    # Roughly 18–55 µT at 500 km altitude.
    assert 15e-6 < magnitude.min() < 25e-6
    assert 45e-6 < magnitude.max() < 60e-6


def test_ecef_vector_matches_spherical_components(igrf: Igrf) -> None:
    r_e = geodetic_to_ecef(math.radians(60.2), math.radians(24.9), 500e3)
    b_ecef = igrf.field_ecef(r_e, 2021.8)
    rad, colat, lon = ecef_to_spherical(r_e)
    b_r, _, _ = igrf.field_spherical(rad, colat, lon, 2021.8)
    radial = r_e / np.linalg.norm(r_e)
    assert float(b_ecef @ radial) == pytest.approx(float(b_r) * 1e-9, rel=1e-12)
    # Northern hemisphere: the field points down and roughly north.
    assert b_ecef @ radial < 0
