"""Simulator behaviour hardened in the 2026-11 review: exact duty cycle, non-finite commands
and states, batched environment evaluation and run metadata."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from adcssim.actuators import Magnetorquers
from adcssim.backends import AdcsInput, AdcsOutput, NoAdcs
from adcssim.dynamics import rk4_step
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, parse_scenario
from tests.conftest import REPO


def scenario(**time: float) -> Scenario:
    return parse_scenario(
        {
            "name": "hardening",
            "time": {"duration_s": 4.0, **time},
            "orbit": {"tle": "data/tle/aalto-1_2021-287.tle"},
            "initial": {"rate_deg_s": [3.0, 5.0, -4.0]},
            "disturbances": {"aerodynamic": False, "solar_pressure": False},
        },
        base_dir=REPO,
    )


class ConstantCommand(NoAdcs):
    """Commands a fixed dipole, to test the simulator independently of the ADCS."""

    name = "constant"

    def __init__(self, m: list[float]) -> None:
        self.m = np.array(m)

    def step(self, inp: AdcsInput) -> AdcsOutput:
        return AdcsOutput(m_cmd=self.m.copy())


def test_nonfinite_command_components_are_zeroed() -> None:
    act = Magnetorquers()
    m = act.apply(np.array([np.nan, 0.05, np.inf]))
    np.testing.assert_array_equal(m, [0.0, 0.05, 0.0])


def test_duty_cycle_splits_the_straddling_substep() -> None:
    # dt 0.1 s, duty 0.25: the coils switch off half-way through the third substep.
    sc = scenario(duration_s=1.0, dt_dynamics_s=0.1)
    sc = replace(sc, actuators=Magnetorquers(duty_cycle=0.25))
    m = [0.1, -0.05, 0.08]
    sim = Simulation(sc, ConstantCommand(m))
    q0, w0 = sim.q.copy(), sim.w.copy()
    sim.step()

    # Reference: the same period integrated with substeps fine enough to hit 0.25 s exactly.
    env = sim._environment(0)
    q, w = q0, w0
    h = 0.05
    for i in range(20):
        j = int(i * h / 0.1 + 1e-9)
        on = i * h < 0.25 - 1e-12
        torque = sim._torque_fn(env, j, np.array(m) if on else np.zeros(3))
        q, w = rk4_step(q, w, h, sc.spacecraft.inertia, torque, sim._inertia_inv)
    np.testing.assert_allclose(sim.w, w, atol=1e-12)
    np.testing.assert_allclose(sim.q, q, atol=1e-12)


def test_nonfinite_truth_state_raises() -> None:
    sim = Simulation(scenario())
    sim.w = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(FloatingPointError, match="non-finite"):
        sim.step()


def test_batched_environment_matches_per_period_evaluation() -> None:
    sc = scenario(duration_s=200.0)
    sim = Simulation(sc)
    n = sc.substeps
    for k in (0, 63, 64, 150, 199):
        batched = sim._environment(k)
        direct = sim.env.evaluate((k * n + np.arange(n)) * sc.dt_dynamics_s)
        np.testing.assert_array_equal(batched.t, direct.t)
        np.testing.assert_array_equal(batched.r, direct.r)
        # one IGRF epoch per batch instead of per period: secular variation is ~1e-5 nT/s
        np.testing.assert_allclose(batched.b, direct.b, rtol=0, atol=1e-12)


def test_run_outside_the_igrf_range_is_rejected() -> None:
    sc = replace(scenario(), start=datetime(2040, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="IGRF"):
        Simulation(sc)


def test_run_metadata_records_config_and_code_version() -> None:
    meta = Simulation(scenario()).metadata()
    assert json.loads(meta["adcs_config"])["variant"] == meta["adcs_variant"]
    assert "git_commit" in meta
    assert meta["library"] == ""  # Python backend
