import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adcssim.backends import AdcsContext, AdcsInput, AdcsOutput, NoAdcs, create_backend
from adcssim.dynamics import inertial_angular_momentum
from adcssim.frames import quat_rotate
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, load_scenario, parse_scenario
from adcssim.telemetry import SCHEMA, Telemetry, column_names
from tests.conftest import REPO


def scenario(**overrides: Any) -> Scenario:
    data: dict[str, Any] = {
        "name": "test",
        "seed": 3,
        "time": {"duration_s": 60.0, "start_offset_s": 3816.0},
        "orbit": {"tle": "data/tle/aalto-1_2021-287.tle"},
        "initial": {"rate_deg_s": [3.0, 5.0, -4.0]},
        "sensors": {
            "magnetometer": {"noise_std_nT": 100.0},
            "gyroscope": {"noise_std_deg_s": 0.01},
            "sun": {"noise_std_deg": 0.2},
        },
    }
    data.update(overrides)
    return parse_scenario(data, base_dir=REPO)


class ConstantDipole:
    """Test backend that always commands the same dipole."""

    name = "constant"

    def __init__(self, m: list[float]) -> None:
        self.m = np.array(m)
        self.inputs: list[AdcsInput] = []
        self.ctx: AdcsContext | None = None

    def reset(self, ctx: AdcsContext) -> None:
        self.ctx = ctx

    def step(self, inp: AdcsInput) -> AdcsOutput:
        self.inputs.append(inp)
        return AdcsOutput(m_cmd=self.m)


def test_run_produces_one_row_per_control_period() -> None:
    tel = Simulation(scenario()).run()
    assert len(tel) == 60
    np.testing.assert_allclose(tel.get("t"), np.arange(60.0))
    for name in SCHEMA:
        for col in column_names(name):
            assert col in tel.columns
    assert tel.metadata["backend"] == "none"
    assert tel.metadata["seed"] == "3"


def test_runs_are_deterministic() -> None:
    a = Simulation(scenario()).run()
    b = Simulation(scenario()).run()
    for col in a.columns:
        if col != "compute_time_s":
            np.testing.assert_array_equal(a.columns[col], b.columns[col], err_msg=col)


def test_seed_changes_noise_only() -> None:
    a = Simulation(scenario()).run()
    b = Simulation(scenario(seed=4)).run()
    np.testing.assert_array_equal(a.get("q_true"), b.get("q_true"))  # open loop: same truth
    assert not np.array_equal(a.get("mag"), b.get("mag"))


def test_logged_truth_is_consistent() -> None:
    tel = Simulation(scenario()).run()
    q, b_i, b_b = tel.get("q_true"), tel.get("b_i"), tel.get("b_b_true")
    np.testing.assert_allclose(quat_rotate(q, b_i), b_b, atol=1e-15)
    # Readings scatter around the truth with the configured noise.
    residual = tel.get("mag") - b_b
    assert np.std(residual) == pytest.approx(100e-9, rel=0.2)


def test_torque_free_run_conserves_angular_momentum() -> None:
    sc = scenario(
        disturbances={
            "gravity_gradient": False,
            "residual_dipole": False,
            "aerodynamic": False,
            "solar_pressure": False,
        },
        time={"duration_s": 600.0},
    )
    tel = Simulation(sc).run()
    j = sc.spacecraft.inertia
    h = np.array(
        [
            inertial_angular_momentum(q, w, j)
            for q, w in zip(tel.get("q_true"), tel.get("w_true"), strict=True)
        ]
    )
    np.testing.assert_allclose(h, np.broadcast_to(h[0], h.shape), rtol=1e-9, atol=1e-15)


def test_disturbances_change_the_motion() -> None:
    quiet = scenario(
        disturbances={
            "gravity_gradient": False,
            "residual_dipole": False,
            "aerodynamic": False,
            "solar_pressure": False,
        }
    )
    a = Simulation(quiet).run()
    b = Simulation(scenario()).run()
    assert not np.allclose(a.get("w_true")[-1], b.get("w_true")[-1], rtol=0, atol=1e-12)
    assert np.all(np.linalg.norm(b.get("tau_disturbance"), axis=1) < 1e-6)


def test_backend_receives_readings_and_drives_the_coils() -> None:
    backend = ConstantDipole([0.1, 0.0, 0.0])
    sc = scenario(time={"duration_s": 30.0, "start_offset_s": 3816.0})
    tel = Simulation(sc, backend).run()
    assert backend.ctx is not None and backend.ctx.dt == 1.0
    assert len(backend.inputs) == 30
    first = backend.inputs[0]
    assert first.mag_valid and first.gyro_valid
    np.testing.assert_allclose(np.linalg.norm(first.sun_ref_i), 1.0)
    np.testing.assert_allclose(tel.get("m_applied"), np.tile([0.1, 0.0, 0.0], (30, 1)))
    np.testing.assert_allclose(
        tel.get("tau_control"), np.cross(tel.get("m_applied"), tel.get("b_b_true")), atol=1e-20
    )
    open_loop = Simulation(sc, NoAdcs()).run()
    assert not np.allclose(tel.get("w_true")[-1], open_loop.get("w_true")[-1])


def test_dipole_command_is_saturated() -> None:
    tel = Simulation(scenario(time={"duration_s": 5.0}), ConstantDipole([1.0, -1.0, 1.0])).run()
    np.testing.assert_allclose(tel.get("m_applied")[0], [0.13, -0.13, 0.082])


def test_duty_cycle_reduces_the_applied_impulse() -> None:
    def momentum_change(duty: float) -> float:
        sc = scenario(
            time={"duration_s": 20.0, "start_offset_s": 3816.0},
            actuators={"duty_cycle": duty},
            disturbances={
                "gravity_gradient": False,
                "residual_dipole": False,
                "aerodynamic": False,
                "solar_pressure": False,
            },
        )
        tel = Simulation(sc, ConstantDipole([0.1, 0.1, 0.05])).run()
        j = sc.spacecraft.inertia
        q, w = tel.get("q_true"), tel.get("w_true")
        h0 = inertial_angular_momentum(q[0], w[0], j)
        h1 = inertial_angular_momentum(q[-1], w[-1], j)
        return float(np.linalg.norm(h1 - h0))

    full, half = momentum_change(1.0), momentum_change(0.5)
    assert half == pytest.approx(0.5 * full, rel=0.05)


def test_eclipse_timing_of_thesis_case() -> None:
    sc = dataclasses.replace(load_scenario(REPO / "scenarios" / "case_1a.toml"), duration_s=1100)
    sim = Simulation(sc)
    entry = None
    for frame in sim.frames():
        if frame.env.eclipse[0]:
            entry = frame.t
            break
    assert entry == pytest.approx(1087.0, abs=2.0)


def test_attitude_error_is_nan_without_estimate() -> None:
    tel = Simulation(scenario(time={"duration_s": 3.0})).run()
    assert np.all(np.isnan(tel.get("attitude_error")))


def test_step_after_end_raises() -> None:
    sim = Simulation(scenario(time={"duration_s": 2.0}))
    sim.run()
    assert sim.done
    with pytest.raises(RuntimeError):
        sim.step()


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown ADCS backend"):
        create_backend("quantum")


def test_telemetry_round_trip(tmp_path: Path) -> None:
    tel = Simulation(scenario(time={"duration_s": 5.0})).run()
    path = tmp_path / "run.parquet"
    tel.write(path)
    back = Telemetry.read(path)
    assert back.metadata == tel.metadata
    assert "scenario_toml" in back.metadata
    for col, values in tel.columns.items():
        np.testing.assert_array_equal(back.columns[col], values, err_msg=col)
    assert back.get("q_true").shape == (5, 4)
    assert back.get("eclipse").dtype == np.bool_
    assert math.isclose(float(back.get("t")[-1]), 4.0)


def test_diagnostic_columns_are_nan_without_a_reference_backend() -> None:
    tel = Simulation(scenario(time={"duration_s": 3.0})).run()
    assert np.all(np.isnan(tel.get("nis"))) and tel.get("innov").shape == (3, 13)


def test_old_logs_without_new_fields_read_as_nan(tmp_path: Path) -> None:
    tel = Simulation(scenario(time={"duration_s": 3.0})).run()
    cols = {k: v for k, v in tel.columns.items() if not k.startswith("nis")}
    old = Telemetry(cols, tel.metadata)
    assert not old.has("nis")
    assert old.get("nis").shape == (3, 4) and np.all(np.isnan(old.get("nis")))
