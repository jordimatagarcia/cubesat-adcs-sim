"""Sensors, actuators, disturbances and spacecraft parameters."""

import math

import numpy as np
import pytest

from adcssim.actuators import FS1_MAX_DIPOLE, Magnetorquers, magnetic_torque
from adcssim.disturbances import (
    DisturbanceModel,
    aerodynamic,
    gravity_gradient,
    residual_dipole,
    solar_pressure,
)
from adcssim.environment import AU
from adcssim.orbit import MU_EARTH_WGS72
from adcssim.sensors import (
    GYRO_LSB_RAD_S,
    MAG_LSB_T,
    BiasWindow,
    GyroscopeModel,
    MagnetometerModel,
    SensorErrors,
    SunSensorModel,
)
from adcssim.spacecraft import Spacecraft, box_inertia

B = np.array([20e-6, -5e-6, 35e-6])


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- spacecraft -----------------------------------------------------------------------


def test_box_inertia_of_3u() -> None:
    j = box_inertia(4.0, (0.1, 0.1, 0.34))
    assert j[0, 0] == pytest.approx(4.0 / 12 * (0.01 + 0.1156))
    assert j[2, 2] == pytest.approx(4.0 / 12 * 0.02)


def test_spacecraft_validates_inertia() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        Spacecraft(inertia=np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="positive"):
        Spacecraft(inertia=np.diag([1.0, -1.0, 1.0]))


def test_projected_area() -> None:
    sc = Spacecraft()
    assert sc.projected_area(np.array([1.0, 0, 0])) == pytest.approx(0.034)
    assert sc.projected_area(np.array([0, 0, 1.0])) == pytest.approx(0.01)


# --- magnetometer ---------------------------------------------------------------------


def test_ideal_magnetometer_returns_truth() -> None:
    mag = MagnetometerModel(errors=SensorErrors())
    reading, valid = mag.measure(0.0, B, rng())
    np.testing.assert_array_equal(reading, B)
    assert valid


def test_magnetometer_noise_statistics() -> None:
    mag = MagnetometerModel(errors=SensorErrors(noise_std=1e-7))
    g = rng(1)
    samples = np.array([mag.measure(0.0, B, g)[0] for _ in range(20_000)])
    np.testing.assert_allclose(samples.std(axis=0), 1e-7, rtol=0.03)
    np.testing.assert_allclose(samples.mean(axis=0), B, atol=5e-9)


def test_magnetometer_bias_window() -> None:
    # Thesis case 2b: b_mag = 1e-5 × [0, −3, 0] T during 125–225 s.
    window = BiasWindow(125.0, 225.0, 1e-5 * np.array([0.0, -3.0, 0.0]))
    mag = MagnetometerModel(errors=SensorErrors(bias_windows=(window,)))
    before, _ = mag.measure(100.0, B, rng())
    during, _ = mag.measure(150.0, B, rng())
    np.testing.assert_allclose(before, B)
    np.testing.assert_allclose(during - B, [0.0, -3e-5, 0.0])
    np.testing.assert_allclose(mag.true_bias(150.0), [0.0, -3e-5, 0.0])


def test_magnetometer_quantisation_and_saturation() -> None:
    mag = MagnetometerModel()  # datasheet range and resolution
    reading, valid = mag.measure(0.0, B, rng())
    np.testing.assert_allclose(reading / MAG_LSB_T, np.round(reading / MAG_LSB_T), atol=1e-6)
    assert valid
    _, valid = mag.measure(0.0, np.array([2e-3, 0.0, 0.0]), rng())
    assert not valid


def test_magnetometer_hard_and_soft_iron() -> None:
    hard = np.array([1e-6, 0.0, -2e-6])
    soft = np.diag([0.01, 0.0, 0.0])
    mag = MagnetometerModel(errors=SensorErrors(), hard_iron=hard, soft_iron=soft)
    reading, _ = mag.measure(0.0, B, rng())
    np.testing.assert_allclose(reading, (np.eye(3) + soft) @ B + hard)
    np.testing.assert_allclose(mag.true_bias(0.0), hard)


def test_scale_factor_and_misalignment() -> None:
    errors = SensorErrors(
        scale_factor=np.array([0.01, 0.0, 0.0]), misalignment=np.array([0.0, 0.0, 1e-3])
    )
    x = np.array([1.0, 0.0, 0.0])
    # K_mis = I − [θ×] with θ = (0, 0, 1e-3) maps x̂ to (1, −1e-3, 0); scale 1.01 on x.
    np.testing.assert_allclose(errors.matrix @ x, [1.01, -1.01e-3, 0.0])


# --- gyroscope ------------------------------------------------------------------------


def test_gyroscope_bias_and_quantisation() -> None:
    bias = np.radians([0.0, 0.0, 5.0])  # thesis case 2a
    gyro = GyroscopeModel(
        errors=SensorErrors(bias_windows=(BiasWindow(75.0, 175.0, bias),), lsb=GYRO_LSB_RAD_S)
    )
    w = np.radians([3.0, 5.0, -4.0])
    reading, _ = gyro.measure(100.0, w, rng())
    np.testing.assert_allclose(reading, w + bias, atol=GYRO_LSB_RAD_S / 2 + 1e-15)


def test_gyroscope_bias_random_walk_grows_with_time() -> None:
    finals = []
    for seed in range(400):
        gyro = GyroscopeModel(errors=SensorErrors(), bias_walk_std=1e-5)
        g = rng(seed)
        for t in np.arange(0.0, 101.0, 1.0):
            gyro.measure(float(t), np.zeros(3), g)
        finals.append(gyro.true_bias(100.0))
    # Random walk: σ(t) = σ_walk √t = 1e-4 rad/s after 100 s.
    np.testing.assert_allclose(np.std(finals, axis=0), 1e-4, rtol=0.12)


def test_gyroscope_rejects_time_going_backwards() -> None:
    gyro = GyroscopeModel(bias_walk_std=1e-5)
    gyro.measure(10.0, np.zeros(3), rng())
    with pytest.raises(ValueError, match="backwards"):
        gyro.measure(5.0, np.zeros(3), rng())


# --- sun sensors ----------------------------------------------------------------------


def test_sun_sensor_invalid_in_eclipse() -> None:
    reading, valid, _ = SunSensorModel().measure(np.array([1.0, 0, 0]), True, rng())
    assert not valid
    np.testing.assert_array_equal(reading, 0.0)


def test_sun_sensor_coverage_gaps() -> None:
    corner = np.ones(3) / math.sqrt(3)  # 54.7° from every face normal
    _, valid, faces = SunSensorModel().measure(corner, False, rng())
    assert not valid and not faces.any()
    _, valid, _ = SunSensorModel(half_fov_rad=math.pi / 2).measure(corner, False, rng())
    assert valid


def test_sun_sensor_angular_noise() -> None:
    sensor = SunSensorModel(noise_std_rad=math.radians(0.5), half_fov_rad=math.pi / 2)
    s = np.array([0.0, 0.6, 0.8])
    g = rng(3)
    angles = []
    for _ in range(5000):
        reading, valid, _ = sensor.measure(s, False, g)
        assert valid
        assert np.linalg.norm(reading) == pytest.approx(1.0)
        angles.append(math.acos(min(1.0, float(reading @ s))))
    # Two transverse axes with σ each: the angle is Rayleigh-distributed, RMS = σ√2.
    rms = math.sqrt(np.mean(np.square(angles)))
    assert rms == pytest.approx(math.radians(0.5) * math.sqrt(2), rel=0.05)


def test_sensors_are_deterministic_for_a_seed() -> None:
    mag = MagnetometerModel(errors=SensorErrors(noise_std=1e-7))
    a = [mag.measure(0.0, B, g)[0] for g in [rng(7)] for _ in range(3)]
    b = [mag.measure(0.0, B, g)[0] for g in [rng(7)] for _ in range(3)]
    np.testing.assert_array_equal(a, b)


# --- actuators ------------------------------------------------------------------------


def test_magnetorquers_clip_and_quantise() -> None:
    mtq = Magnetorquers(levels=100)
    m = mtq.apply(np.array([1.0, -0.05, 0.0123]))
    assert m[0] == pytest.approx(FS1_MAX_DIPOLE[0])
    assert m[1] == pytest.approx(-0.05, abs=0.13 / 200)
    assert m[2] == pytest.approx(0.0123, abs=0.082 / 200)


def test_magnetorquer_duty_cycle() -> None:
    mtq = Magnetorquers(duty_cycle=0.8)
    assert mtq.active(0.0, 1.0)
    assert mtq.active(0.79, 1.0)
    assert not mtq.active(0.8, 1.0)
    with pytest.raises(ValueError):
        Magnetorquers(duty_cycle=0.0)


def test_magnetic_torque_is_perpendicular() -> None:
    tau = magnetic_torque(np.array([0.1, 0.0, 0.0]), B)
    assert float(tau @ B) == pytest.approx(0.0, abs=1e-20)


# --- disturbances ---------------------------------------------------------------------


def test_gravity_gradient_torque() -> None:
    i1, i3 = 0.0418, 0.0067
    j = np.diag([i1, i1, i3])
    r = 6.878e6
    # Aligned with a principal axis: no torque.
    np.testing.assert_allclose(gravity_gradient(np.array([0, 0, r]), j), 0.0, atol=1e-20)
    # 45° between the long axis and nadir: τ_y = 3μ/(2r³) (I1 − I3), the maximum.
    r_b = r * np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
    tau = gravity_gradient(r_b, j)
    assert tau[1] == pytest.approx(3 * MU_EARTH_WGS72 / (2 * r**3) * (i1 - i3))
    assert tau[0] == pytest.approx(0.0, abs=1e-20)


def test_residual_dipole_torque() -> None:
    np.testing.assert_allclose(
        residual_dipole(np.array([1e-3, 0, 0]), np.array([0, 3e-5, 0])), [0, 0, 3e-8]
    )


def test_pressure_torques_vanish_at_centre_of_mass() -> None:
    sc = Spacecraft(centre_of_pressure=np.zeros(3))
    np.testing.assert_allclose(aerodynamic(sc, np.array([7.6e3, 0, 0]), 7e-13), 0.0)
    np.testing.assert_allclose(solar_pressure(sc, np.array([0, 0, 1.0]), AU, False), 0.0)


def test_drag_torque_direction() -> None:
    sc = Spacecraft(centre_of_pressure=np.array([0.0, 0.0, 0.01]))
    tau = aerodynamic(sc, np.array([7.6e3, 0.0, 0.0]), 7e-13)
    # Drag along −x applied at +z gives a torque about −y.
    force = 0.5 * 7e-13 * 2.2 * 0.034 * 7.6e3**2
    np.testing.assert_allclose(tau, [0.0, -0.01 * force, 0.0], rtol=1e-12)


def test_no_solar_pressure_in_eclipse() -> None:
    sc = Spacecraft()
    np.testing.assert_array_equal(solar_pressure(sc, np.array([0, 1.0, 0]), AU, True), 0.0)


def test_disturbance_model_switches() -> None:
    sc = Spacecraft()
    args = (
        sc,
        np.array([5e6, 3e6, 3e6]),
        B,
        np.array([7e3, 0, 0]),
        1e-12,
        np.array([0, 1.0, 0]),
        AU,
        False,
    )
    off = DisturbanceModel(False, False, False, False).torque(*args)
    np.testing.assert_array_equal(off, 0.0)
    only_gg = DisturbanceModel(True, False, False, False).torque(*args)
    np.testing.assert_allclose(only_gg, gravity_gradient(args[1], sc.inertia))
