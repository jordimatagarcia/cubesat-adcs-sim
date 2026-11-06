"""Noise settings of the improved variant matched to a scenario's sensor models.

The filter's measurement noise has to cover both the sensor noise and the error of the
on-board reference models (the IGRF field and the Sun ephemeris the ADCS compares the
readings with). Those models are good to roughly 0.1 µT and 0.01°, so the filter never
assumes better than that, however good the sensor: with the thesis's near-ideal sensors
(σ = 1 pT for the magnetometer) an unfloored filter would trust the magnetometer so
absolutely that a bias step drags the attitude with it.
"""

import math
from dataclasses import replace

from adcssim.backends.config import AdcsConfig
from adcssim.scenario import Scenario

MAG_FLOOR_UT = 0.1
SUN_FLOOR_RAD = 2e-4
GYRO_FLOOR_RAD_S = 1e-5


def matched_improved_config(scenario: Scenario, variant: str = "improved") -> AdcsConfig:
    """The scenario's ADCS configuration switched to the improved (or ``mekf``) variant,
    with the sensor noise taken from the scenario's sensor models (and the floors above)."""
    sensors = scenario.make_sensors()
    mag = max(sensors.magnetometer.errors.noise_std * 1e6, MAG_FLOOR_UT)
    sun = max(sensors.sun.noise_std_rad, SUN_FLOOR_RAD)
    gyro = max(sensors.gyroscope.errors.noise_std, GYRO_FLOOR_RAD_S)
    # Quantisation adds uniform noise of variance lsb²/12.
    if sensors.gyroscope.errors.lsb > 0.0:
        gyro = math.hypot(gyro, sensors.gyroscope.errors.lsb / math.sqrt(12.0))
    if sensors.magnetometer.errors.lsb > 0.0:
        mag = math.hypot(mag, sensors.magnetometer.errors.lsb * 1e6 / math.sqrt(12.0))
    return replace(
        scenario.adcs,
        variant=variant,
        sigma_mag_noise=mag,
        sigma_sun_noise=sun,
        sigma_gyro_noise=gyro,
    )
