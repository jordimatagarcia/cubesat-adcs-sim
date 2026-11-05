import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adcssim.environment import (
    AU,
    R_EARTH,
    Environment,
    atmospheric_density,
    in_eclipse,
    sun_position,
)
from adcssim.frames import unit
from adcssim.orbit import Orbit, Tle
from adcssim.timebase import Clock, julian_date


def test_sun_direction_matches_astropy(astropy_ref: list[dict[str, Any]]) -> None:
    for point in astropy_ref:
        t = datetime.fromisoformat(point["utc"]).replace(tzinfo=UTC)
        ours = unit(sun_position(*julian_date(t)))
        ref = unit(point["sun_teme_m"])
        angle = math.degrees(math.acos(min(1.0, float(ours @ ref))))
        assert angle < 0.01, point["utc"]
        assert np.linalg.norm(sun_position(*julian_date(t))) == pytest.approx(
            np.linalg.norm(point["sun_teme_m"]), rel=2e-4
        )


def test_sun_distance_range() -> None:
    jd, fr = julian_date(datetime(2021, 1, 1, tzinfo=UTC))
    d = np.linalg.norm(sun_position(jd, fr + np.arange(0, 366)), axis=-1) / AU
    assert d.min() == pytest.approx(0.9833, abs=1e-3)  # perihelion, early January
    assert d.max() == pytest.approx(1.0167, abs=1e-3)  # aphelion, early July


def test_eclipse_geometry() -> None:
    sun = np.array([AU, 0.0, 0.0])
    r = R_EARTH + 500e3
    assert in_eclipse([-r, 0.0, 0.0], sun)  # directly behind the Earth
    assert not in_eclipse([r, 0.0, 0.0], sun)  # between Earth and Sun
    assert not in_eclipse([-r, 0.0, R_EARTH + 1.0], sun)  # behind but off the shadow
    assert in_eclipse([-r, 0.0, R_EARTH - 1.0], sun)
    assert not in_eclipse([0.0, r, 0.0], sun)  # terminator


def test_eclipse_fraction_matches_analytic_value(aalto1_tle_path: Path) -> None:
    orbit = Orbit.from_tle(Tle.from_file(aalto1_tle_path))
    clock = Clock(orbit.epoch)
    env = Environment(orbit, clock)
    t = np.linspace(0.0, orbit.period_s, 4000, endpoint=False)
    sample = env.evaluate(t)
    fraction = float(np.mean(sample.eclipse))

    # Cylindrical shadow, circular orbit: f = arccos(√(h² + 2Rh) / ((R + h) cos β)) / π,
    # with β the angle between the Sun and the orbit plane.
    radius = float(np.mean(np.linalg.norm(sample.r, axis=1)))
    normal = unit(np.cross(sample.r[0], sample.v[0]))
    beta = math.asin(float(sample.sun[0] @ normal))
    h = radius - R_EARTH
    expected = math.acos(math.sqrt(h * h + 2 * R_EARTH * h) / (radius * math.cos(beta))) / math.pi
    assert fraction == pytest.approx(expected, abs=0.01)


def test_density() -> None:
    assert float(atmospheric_density(500e3)) == pytest.approx(6.967e-13)
    h = np.linspace(100e3, 1000e3, 200)
    assert np.all(np.diff(atmospheric_density(h)) < 0)


def test_environment_sample(aalto1_tle_path: Path) -> None:
    orbit = Orbit.from_tle(Tle.from_file(aalto1_tle_path))
    env = Environment(orbit, Clock(orbit.epoch))
    s = env.evaluate(np.arange(0.0, 60.0, 0.1))
    assert s.r.shape == s.b.shape == s.sun.shape == (600, 3)
    np.testing.assert_allclose(np.linalg.norm(s.sun, axis=1), 1.0)
    assert np.all((np.linalg.norm(s.b, axis=1) > 15e-6) & (np.linalg.norm(s.b, axis=1) < 60e-6))
    assert np.all((s.alt > 480e3) & (s.alt < 530e3))
    # The co-rotating atmosphere removes up to ~0.5 km/s of the orbital speed.
    dv = np.linalg.norm(s.v, axis=1) - np.linalg.norm(s.v_rel, axis=1)
    assert np.all(np.abs(dv) < 600.0)
    one = s.at(10)
    assert one.r.shape == (1, 3)
    np.testing.assert_array_equal(one.r[0], s.r[10])
