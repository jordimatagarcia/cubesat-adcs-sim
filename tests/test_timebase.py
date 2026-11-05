import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from sgp4.propagation import gstime

from adcssim.timebase import Clock, decimal_year, from_julian_date, gmst, julian_date


def test_julian_date_of_j2000() -> None:
    jd, fr = julian_date(datetime(2000, 1, 1, 12, tzinfo=UTC))
    assert jd + fr == 2_451_545.0


def test_julian_date_vallado_example_3_4() -> None:
    # Vallado, Fundamentals of Astrodynamics, example 3-4: 1996-10-26 14:20:00 UT.
    jd, fr = julian_date(datetime(1996, 10, 26, 14, 20, tzinfo=UTC))
    assert jd + fr == pytest.approx(2_450_383.097_222_22, abs=1e-8)


def test_julian_date_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone"):
        julian_date(datetime(2021, 1, 1))


def test_julian_date_round_trip() -> None:
    t = datetime(2021, 10, 14, 13, 9, 39, 745_152, tzinfo=UTC)
    back = from_julian_date(*julian_date(t))
    assert abs((back - t).total_seconds()) < 1e-5


def test_decimal_year() -> None:
    assert decimal_year(datetime(2021, 1, 1, tzinfo=UTC)) == 2021.0
    assert decimal_year(datetime(2021, 7, 2, 12, tzinfo=UTC)) == pytest.approx(2021.5)


def test_gmst_vallado_example_3_5() -> None:
    # Vallado example 3-5: 1992-08-20 12:14:00 UT1 → GMST = 152.578787810°.
    jd, fr = julian_date(datetime(1992, 8, 20, 12, 14, tzinfo=UTC))
    assert math.degrees(float(gmst(jd, fr))) == pytest.approx(152.578_787_810, abs=1e-6)


def test_gmst_matches_astropy(astropy_ref: list[dict[str, Any]]) -> None:
    for point in astropy_ref:
        t = datetime.fromisoformat(point["utc"]).replace(tzinfo=UTC)
        diff = float(gmst(*julian_date(t))) - point["gmst_rad"]
        assert abs(math.remainder(diff, 2 * math.pi)) < 1e-9, point["utc"]


def test_gmst_matches_sgp4_package() -> None:
    for days in np.linspace(0, 9000, 17):
        t = datetime(2001, 3, 4, 5, 6, 7, tzinfo=UTC) + timedelta(days=float(days))
        jd, fr = julian_date(t)
        assert float(gmst(jd, fr)) == pytest.approx(gstime(jd + fr), abs=1e-9)


def test_gmst_is_vectorised_and_in_range() -> None:
    jd, fr = julian_date(datetime(2021, 10, 14, tzinfo=UTC))
    g = gmst(jd, fr + np.linspace(0, 2, 1000))
    assert g.shape == (1000,)
    assert np.all((g >= 0) & (g < 2 * math.pi))


def test_clock_maps_simulation_time() -> None:
    start = datetime(2021, 10, 14, 13, 0, tzinfo=UTC)
    clock = Clock(start)
    jd, fr = clock.jd(3600.0)
    jd0, fr0 = julian_date(start)
    # Compare the split parts: a summed Julian date only resolves ~5e-10 days.
    assert jd == jd0
    assert float(fr) == pytest.approx(fr0 + 1 / 24, abs=1e-15)
    assert clock.datetime(90.0) == start + timedelta(seconds=90)
