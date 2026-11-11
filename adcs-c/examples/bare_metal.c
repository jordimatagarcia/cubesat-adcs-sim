/*
 * bare_metal.c: the ADCS on a microcontroller, without an OS or a heap.
 *
 * Built in CI for an ARM Cortex-M4F (single precision) and a Cortex-M7 (double precision)
 * to check that the library compiles and links for bare metal, and for the host, where ctest
 * runs it (exit code 0: the state fits and the ADCS runs) against the C library and the Rust
 * library alike. The sensor reads and the coil driver are stubs.
 */
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "adcs.h"

/* The state lives in static memory; adcs_state_size() is checked against this at start.
 * 48 KB covers both libraries in double precision (C 36 KB, Rust 42 KB). */
#define STATE_BYTES 48000u
static alignas(16) uint8_t state_memory[STATE_BYTES];

static adcs_config make_config(void)
{
    adcs_config c = {0};
    c.variant = ADCS_VARIANT_MEKF;
    c.dt = (adcs_real)0.1;
    c.process_dt = (adcs_real)0.1;
    c.inertia[0] = c.inertia[4] = (adcs_real)0.0419;
    c.inertia[8] = (adcs_real)0.0067;
    c.max_dipole[0] = c.max_dipole[1] = (adcs_real)0.13;
    c.max_dipole[2] = (adcs_real)0.082;
    c.flae_weight_sun = c.flae_weight_mag = (adcs_real)1;
    c.flae_tolerance = (adcs_real)1e-3;
    c.flae_max_iterations = 20;
    c.sigma_gyro_noise = (adcs_real)1e-4;
    c.sigma_gyro_bias_walk = (adcs_real)3e-6;
    c.sigma_mag_noise = (adcs_real)0.3;
    c.sigma_mag_bias_walk = (adcs_real)1e-3;
    c.sigma_sun_noise = (adcs_real)5e-3;
    c.p0_attitude = (adcs_real)0.05;
    c.p0_gyro_bias = (adcs_real)5e-3;
    c.p0_mag_bias = (adcs_real)2;
    c.fault_nis_threshold = (adcs_real)25;
    c.fault_steps = 3;
    c.fault_gyro_bias_sigma = (adcs_real)0.1;
    c.fault_mag_bias_sigma = (adcs_real)50;
    c.fault_attitude_sigma = (adcs_real)0.05;
    c.sigma_outage_accel = (adcs_real)1e-3;
    c.control_enabled = true;
    c.bdot_gain = (adcs_real)4e4;
    c.kp = (adcs_real)2e-5;
    c.kd = (adcs_real)2e-3;
    c.q_target[3] = (adcs_real)1;
    c.detumble_enter_rate = (adcs_real)0.07;
    c.detumble_exit_rate = (adcs_real)0.0175;
    c.settle_time = (adcs_real)60;
    return c;
}

/* Board support stubs */
static void read_sensors(adcs_input *in)
{
    static const adcs_input sample = {
        .mag_b = {(adcs_real)-1.45e-5, (adcs_real)2.71e-5, (adcs_real)1.66e-5},
        .mag_valid = true,
        .gyro_b = {(adcs_real)0.0524, (adcs_real)0.0873, (adcs_real)-0.0698},
        .gyro_valid = true,
        .sun_b = {(adcs_real)0.3411, (adcs_real)0.5305, (adcs_real)0.7761},
        .sun_valid = true,
        .sun_ref_i = {(adcs_real)-0.9307, (adcs_real)-0.3356, (adcs_real)-0.1454},
        .mag_ref_i = {(adcs_real)-2.52e-5, (adcs_real)1.24e-5, (adcs_real)-2.08e-5},
    };
    *in = sample;
}

static volatile adcs_real coil_command[3];

static void drive_coils(const adcs_real m[3])
{
    for (int i = 0; i < 3; i++) {
        coil_command[i] = m[i];
    }
}

int main(void)
{
    adcs_state *state = (adcs_state *)(void *)state_memory;
    const adcs_config cfg = make_config();
    if (!adcs_check_abi() || adcs_state_size() > STATE_BYTES || adcs_init(state, &cfg) != 0) {
        return 1;
    }
    for (uint32_t k = 0; k < 100; k++) { /* the control loop, one step per timer tick */
        adcs_input in;
        adcs_output out;
        read_sensors(&in);
        in.t = (adcs_real)k * cfg.dt;
        if (adcs_step(state, &in, &out) != 0) {
            return 1;
        }
        drive_coils(out.m_cmd);
    }
    return 0;
}
