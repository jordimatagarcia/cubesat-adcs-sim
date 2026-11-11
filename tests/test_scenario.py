import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adcssim.scenario import Scenario, ScenarioError, load_scenario, parse_scenario
from tests.conftest import REPO

SCENARIOS = sorted((REPO / "scenarios").glob("*.toml"))


def minimal(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "time": {"duration_s": 10.0},
        "orbit": {"tle": "data/tle/aalto-1_2021-287.tle"},
    }
    data.update(overrides)
    return data


def parse(data: dict[str, Any]) -> Scenario:
    return parse_scenario(data, base_dir=REPO)


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_shipped_scenarios_load(path: Path) -> None:
    sc = load_scenario(path)
    assert sc.name == path.stem
    assert sc.steps * sc.dt_adcs_s == pytest.approx(sc.duration_s)
    assert sc.backend == "python-ref"


def test_thesis_case_values_are_converted_to_si() -> None:
    sc = load_scenario(REPO / "scenarios" / "case_2c.toml")
    np.testing.assert_allclose(sc.w0, np.radians([3.0, 5.0, -4.0]))
    assert sc.dt_adcs_s == pytest.approx(0.1)
    sensors = sc.make_sensors()
    assert sensors.magnetometer.errors.noise_std == pytest.approx(1e-12)  # σ = 1e-12 T
    assert sensors.magnetometer.errors.lsb == 0.0  # quantize = false
    window = sensors.magnetometer.errors.bias_windows[0]
    np.testing.assert_allclose(window.bias, [0.0, -3e-5, 0.0])
    assert sensors.gyroscope.errors.noise_std == pytest.approx(5e-7, rel=1e-5)
    gyro_window = sensors.gyroscope.errors.bias_windows[0]
    np.testing.assert_allclose(gyro_window.bias, np.radians([0.0, 0.0, 5.0]))
    assert sensors.sun.half_fov_rad == pytest.approx(math.pi / 2)
    assert sensors.sun.noise_std_rad == pytest.approx(1e-4, rel=1e-4)
    assert sc.start == sc.orbit.epoch + timedelta(seconds=3816)


def test_adcs_table_overrides_the_thesis_tuning() -> None:
    sc = parse(minimal(adcs={"backend": "python-ref", "v": [1, 2, 3, 4], "kp": 1e-4}))
    assert sc.backend == "python-ref"
    np.testing.assert_array_equal(sc.adcs.v, np.repeat([1.0, 2.0, 3.0, 4.0], [4, 3, 3, 3]))
    assert sc.adcs.kp == 1e-4
    np.testing.assert_array_equal(sc.adcs.w_b, np.repeat([1e-11, 1e-6], [3, 3]))
    sc2 = parse(minimal(adcs={"detumble_exit_rate_deg_s": 0.5}))
    assert sc2.adcs.detumble_exit_rate == pytest.approx(math.radians(0.5))


def test_defaults() -> None:
    sc = parse(minimal())
    np.testing.assert_array_equal(sc.q0, [0, 0, 0, 1])
    assert sc.dt_dynamics_s == 0.1 and sc.dt_adcs_s == 1.0
    assert sc.substeps == 10
    assert sc.backend == "none"
    sensors = sc.make_sensors()
    assert sensors.magnetometer.errors.lsb > 0.0  # datasheet quantisation by default


def test_make_sensors_returns_fresh_models() -> None:
    sc = parse(minimal())
    assert sc.make_sensors().gyroscope is not sc.make_sensors().gyroscope


def test_orbit_from_elements_and_explicit_start() -> None:
    sc = parse(
        minimal(
            orbit={
                "elements": {
                    "epoch": "2026-03-20T00:00:00Z",
                    "altitude_km": 550.0,
                    "inclination_deg": 97.6,
                    "raan_deg": 120.0,
                }
            },
            time={"start": "2026-03-20T06:00:00+02:00", "duration_s": 60.0},
        )
    )
    assert sc.start == datetime(2026, 3, 20, 4, 0, tzinfo=UTC)
    assert sc.orbit.period_s / 60 == pytest.approx(95.6, abs=0.2)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (minimal(nmae="typo"), "unknown key"),
        (minimal(sensors={"magnetometer": {"noise_std_T": 1.0}}), "sensors.magnetometer"),
        (minimal(time={"duration_s": 10.0, "dt_adcs_s": 0.25, "dt_dynamics_s": 0.1}), "multiple"),
        (minimal(time={"duration_s": -1.0}), "positive"),
        ({"time": {"duration_s": 1.0}, "orbit": {"tle": "x"}}, "name"),
        (minimal(orbit={}), "exactly one"),
        (minimal(initial={"rate_deg_s": [1.0, 2.0]}), "3 values"),
        (minimal(initial={"rate_deg_s": "fast"}), "list of numbers"),
        (minimal(time={"duration_s": 1.0, "start": "2026-01-01T00:00:00"}), "time zone"),
        (minimal(spacecraft={"principal_inertia_kg_m2": [1.0, -1.0, 1.0]}), "positive"),
        (minimal(actuators={"duty_cycle": 1.5}), "duty_cycle"),
        (minimal(disturbances={"aerodynamic": "yes"}), "true or false"),
        (minimal(seed=-3), "seed"),
        (minimal(adcs={"kpp": 1.0}), "adcs.kpp"),
        (minimal(adcs={"v": [1.0, 2.0]}), "group values"),
        (minimal(adcs={"q_target": [1.0, 1.0, 0.0, 0.0]}), "unit quaternion"),
        (minimal(time={"duration_s": 10.5}), "whole number of control periods"),
        (minimal(time={"duration_s": 0.5}), "whole number of control periods"),
        (minimal(time={"duration_s": 10.0, "dt_dynamics_s": 2.0}), "multiple"),
        (minimal(time={"duration_s": float("inf")}), "finite"),
        (minimal(name="../escape"), "name"),
        (minimal(initial={"quaternion": [0.0, 0.0, 0.0, 0.0]}), "must not be zero"),
        (minimal(initial={"rate_deg_s": [1.0, float("nan"), 0.0]}), "finite"),
        (minimal(spacecraft={"mass_kg": 0.0}), "mass_kg"),
        (minimal(spacecraft={"drag_coefficient": -2.2}), "negative"),
        (minimal(sensors={"magnetometer": {"noise_std_nT": -1.0}}), "negative"),
        (minimal(sensors={"gyroscope": {"bias_walk_deg_s_sqrt_s": -1.0}}), "negative"),
        (minimal(sensors={"sun": {"half_fov_deg": 0.0}}), "half_fov_deg"),
        (minimal(sensors={"sun": {"half_fov_deg": 200.0}}), "half_fov_deg"),
    ],
)
def test_invalid_scenarios_are_rejected(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        parse(data)


def test_load_reports_file_name(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n[time\n', encoding="utf-8")
    with pytest.raises(ScenarioError, match=r"bad\.toml"):
        load_scenario(bad)


def test_mass_without_inertia_gives_a_uniform_box() -> None:
    light = parse(minimal(spacecraft={"mass_kg": 2.0}))
    default = parse(minimal())
    np.testing.assert_allclose(light.spacecraft.inertia, 0.5 * default.spacecraft.inertia)


def test_duration_override_is_validated() -> None:
    sc = parse(minimal())
    with pytest.raises(ScenarioError, match="whole number"):
        replace(sc, duration_s=10.3)
