import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from adcssim.orbit import Orbit, OrbitError, Tle, tle_checksum
from adcssim.timebase import Clock, julian_date

# Vallado et al. (2006) SGP4 verification set, satellite 00005, and its published state at
# t = 0 min (km, km/s) from the SGP4-VER output.
VER_LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
VER_LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
VER_R0_KM = (7022.46529266, -1400.08296755, 0.03995155)
VER_V0_KM_S = (1.893841015, 6.405893759, 4.534807250)


def test_vallado_verification_case() -> None:
    orbit = Orbit.from_tle(Tle(VER_LINE1, VER_LINE2))
    r, v = orbit.state(*julian_date(orbit.epoch))
    np.testing.assert_allclose(r / 1e3, VER_R0_KM, atol=1e-4)
    np.testing.assert_allclose(v / 1e3, VER_V0_KM_S, atol=1e-7)


def test_checksum() -> None:
    assert tle_checksum(VER_LINE1) == 3
    assert tle_checksum(VER_LINE2) == 7


def test_tle_rejects_bad_checksum() -> None:
    bad = VER_LINE1[:68] + "0"
    with pytest.raises(ValueError, match="checksum"):
        Tle(bad, VER_LINE2)


def test_tle_rejects_malformed_line() -> None:
    with pytest.raises(ValueError, match="malformed"):
        Tle(VER_LINE1[:60], VER_LINE2)


def test_tle_file_with_name_and_comments(aalto1_tle_path: Path) -> None:
    tle = Tle.from_file(aalto1_tle_path)
    assert tle.name == "AALTO-1"
    orbit = Orbit.from_tle(tle)
    expected_epoch = datetime(2021, 10, 14, 13, 9, 39, 745_152, tzinfo=UTC)
    assert abs((orbit.epoch - expected_epoch).total_seconds()) < 1e-3
    assert orbit.period_s / 60 == pytest.approx(94.545, abs=1e-3)
    r, _ = orbit.state(*julian_date(orbit.epoch))
    assert 490e3 < np.linalg.norm(r) - 6_378_137.0 < 520e3


def test_from_elements_circular_orbit() -> None:
    epoch = datetime(2021, 10, 14, tzinfo=UTC)
    orbit = Orbit.from_elements(epoch, 500e3, inclination_deg=97.4, raan_deg=10.0)
    clock = Clock(epoch)
    t, r = orbit.predict(clock, 0.0, orbit.period_s, step_s=60.0)
    radius = np.linalg.norm(r, axis=1)
    # SGP4 short-period terms make a "circular" mean orbit vary by a few km.
    assert np.ptp(radius) < 20e3
    assert np.mean(radius) == pytest.approx(6_378_135.0 + 500e3, abs=15e3)
    h = np.cross(r[0], r[5])
    inclination = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
    assert inclination == pytest.approx(97.4, abs=0.1)
    assert t[0] == 0.0


def test_vectorised_states_match_single_calls(aalto1_tle_path: Path) -> None:
    orbit = Orbit.from_tle(Tle.from_file(aalto1_tle_path))
    jd, fr = julian_date(orbit.epoch)
    frs = fr + np.array([0.0, 0.01, 0.2])
    r_all, v_all = orbit.states(jd, frs)
    for i, f in enumerate(frs):
        r, v = orbit.state(jd, float(f))
        np.testing.assert_allclose(r_all[i], r)
        np.testing.assert_allclose(v_all[i], v)


def test_predicted_orbit_closes(aalto1_tle_path: Path) -> None:
    orbit = Orbit.from_tle(Tle.from_file(aalto1_tle_path))
    clock = Clock(orbit.epoch)
    _, r = orbit.predict(clock, 0.0, orbit.period_s, step_s=orbit.period_s / 200)
    # After one period the satellite is back near its start (J2 drift is a few tens of km).
    assert np.linalg.norm(r[-1] - r[0]) < 100e3


def test_decayed_orbit_raises() -> None:
    epoch = datetime(2021, 1, 1, tzinfo=UTC)
    orbit = Orbit.from_elements(epoch, -100e3, inclination_deg=50.0, raan_deg=0.0)
    with pytest.raises(OrbitError):
        orbit.state(*julian_date(epoch))


@pytest.mark.parametrize("kind", ["tle", "elements"])
def test_orbit_survives_pickling(kind: str, aalto1_tle_path: Path) -> None:
    import pickle

    if kind == "tle":
        orbit = Orbit.from_tle(Tle.from_file(aalto1_tle_path))
    else:
        orbit = Orbit.from_elements(datetime(2021, 1, 1, tzinfo=UTC), 500e3, 97.4, 10.0)
    clone = pickle.loads(pickle.dumps(orbit))
    jd, fr = julian_date(orbit.epoch)
    np.testing.assert_array_equal(clone.state(jd, fr + 0.3)[0], orbit.state(jd, fr + 0.3)[0])
