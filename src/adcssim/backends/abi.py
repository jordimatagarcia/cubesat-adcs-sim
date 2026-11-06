"""ctypes mirror of ``adcs-c/include/adcs.h`` (double-precision build).

Used by the C and Rust backends, which load a shared library exporting the adcs.h functions.
``tests/test_abi.py`` compiles a small C program against the real header and checks that
every struct size and field offset here matches.
"""

import ctypes
from ctypes import Structure, c_bool, c_double, c_uint32

import numpy as np

from adcssim.backends.base import AdcsInput, AdcsMode, AdcsOutput
from adcssim.backends.config import NB, NX, AdcsConfig

ABI_VERSION = 2

Real = c_double
Vec3 = Real * 3
Vec4 = Real * 4


class AdcsConfigC(Structure):
    _fields_ = [
        ("dt", Real),
        ("process_dt", Real),
        ("inertia", Real * 9),
        ("max_dipole", Vec3),
        ("flae_weight_sun", Real),
        ("flae_weight_mag", Real),
        ("flae_tolerance", Real),
        ("flae_max_iterations", c_uint32),
        ("ukf_alpha", Real),
        ("p_x0", Real * NX),
        ("p_b0", Real * NB),
        ("w_x", Real * NX),
        ("w_b", Real * NB),
        ("v", Real * NX),
        ("control_enabled", c_bool),
        ("bdot_gain", Real),
        ("kp", Real),
        ("kd", Real),
        ("q_target", Vec4),
        ("detumble_enter_rate", Real),
        ("detumble_exit_rate", Real),
        ("settle_time", Real),
    ]


class AdcsInputC(Structure):
    _fields_ = [
        ("t", Real),
        ("mag_b", Vec3),
        ("mag_valid", c_bool),
        ("gyro_b", Vec3),
        ("gyro_valid", c_bool),
        ("sun_b", Vec3),
        ("sun_valid", c_bool),
        ("sun_ref_i", Vec3),
        ("mag_ref_i", Vec3),
    ]


class AdcsOutputC(Structure):
    _fields_ = [
        ("q_est", Vec4),
        ("w_est", Vec3),
        ("bias_mag", Vec3),
        ("bias_gyro", Vec3),
        ("p_diag", Real * (NX + NB)),
        ("m_cmd", Vec3),
        ("mode", c_uint32),
        ("status", c_uint32),
    ]


STRUCTS: dict[str, type[Structure]] = {
    "adcs_config": AdcsConfigC,
    "adcs_input": AdcsInputC,
    "adcs_output": AdcsOutputC,
}


def _arr(ctype: type, values: object) -> ctypes.Array:  # type: ignore[type-arg]
    data = np.asarray(values, dtype=np.float64).ravel()
    return ctype(*data)  # type: ignore[no-any-return]


def config_to_c(
    cfg: AdcsConfig, dt: float, inertia: np.ndarray, max_dipole: np.ndarray
) -> AdcsConfigC:
    return AdcsConfigC(
        dt=dt,
        process_dt=cfg.process_dt,
        inertia=_arr(Real * 9, inertia),
        max_dipole=_arr(Vec3, max_dipole),
        flae_weight_sun=cfg.flae_weight_sun,
        flae_weight_mag=cfg.flae_weight_mag,
        flae_tolerance=cfg.flae_tolerance,
        flae_max_iterations=cfg.flae_max_iterations,
        ukf_alpha=cfg.ukf_alpha,
        p_x0=_arr(Real * NX, cfg.p_x0),
        p_b0=_arr(Real * NB, cfg.p_b0),
        w_x=_arr(Real * NX, cfg.w_x),
        w_b=_arr(Real * NB, cfg.w_b),
        v=_arr(Real * NX, cfg.v),
        control_enabled=cfg.control_enabled,
        bdot_gain=cfg.bdot_gain,
        kp=cfg.kp,
        kd=cfg.kd,
        q_target=_arr(Vec4, cfg.q_target),
        detumble_enter_rate=cfg.detumble_enter_rate,
        detumble_exit_rate=cfg.detumble_exit_rate,
        settle_time=cfg.settle_time,
    )


def input_to_c(inp: AdcsInput) -> AdcsInputC:
    return AdcsInputC(
        t=inp.t,
        mag_b=_arr(Vec3, inp.mag_b),
        mag_valid=inp.mag_valid,
        gyro_b=_arr(Vec3, inp.gyro_b),
        gyro_valid=inp.gyro_valid,
        sun_b=_arr(Vec3, inp.sun_b),
        sun_valid=inp.sun_valid,
        sun_ref_i=_arr(Vec3, inp.sun_ref_i),
        mag_ref_i=_arr(Vec3, inp.mag_ref_i),
    )


def output_from_c(out: AdcsOutputC) -> AdcsOutput:
    def a(x: ctypes.Array) -> np.ndarray:  # type: ignore[type-arg]
        return np.array(x[:], dtype=np.float64)

    return AdcsOutput(
        q_est=a(out.q_est),
        w_est=a(out.w_est),
        bias_mag=a(out.bias_mag),
        bias_gyro=a(out.bias_gyro),
        p_diag=a(out.p_diag),
        m_cmd=a(out.m_cmd),
        mode=AdcsMode(out.mode),
        status=int(out.status),
    )
