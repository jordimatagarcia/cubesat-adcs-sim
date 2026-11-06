/*
 * libFuzzer harness for adcs_step.
 *
 * Starts from a realistic configuration and input (golden vector "nominal", step 0) and lets
 * the fuzzer change them. Input format: byte 0 selects the variant (bit 0) and control
 * (bit 1); then records of 9 bytes, one per step: an op byte whose value mod 20 picks an
 * input field to overwrite with the following 8 bytes as a raw double (19 = no change),
 * after which adcs_step runs once. Fields keep their values between steps, so the fuzzer can
 * build up sequences (valid start, then NaN, then huge values, ...).
 *
 * Besides the sanitizers, checks that every step succeeds, the mode and status are in range,
 * the attitude is either unit-norm or all NaN, and the dipole command is finite and within
 * the coil limits.
 */
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "adcs.h"

#define STATUS_MASK 0x7Fu

static const double BASE_INPUT[19] = {
    0.0,  -1.450141663940126e-05, 2.7086103377968737e-05, 1.659670587303675e-05, 1.0,
    0.05236112040037929, 0.08726701557205724, -0.06981379795273616, 1.0,
    0.3410674516387744,  0.530452642799764,   0.77607537467657,     1.0,
    -0.9307017553237238, -0.3356200571876488, -0.14544215293613677,
    -2.5196175932063603e-05, 1.2400784858588448e-05, -2.0755046448532975e-05,
};

static adcs_config base_config(uint32_t variant, bool control)
{
    static const adcs_real p_x0[ADCS_NX] = {1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2,
                                            1e-7, 1e-7, 1e-7, 5e-7, 5e-7, 5e-7};
    static const adcs_real w_x[ADCS_NX] = {1e-8, 1e-8, 1e-8, 1e-8, 1e-4, 1e-4, 1e-4,
                                           2e-2, 2e-2, 2e-2, 1e-9, 1e-9, 1e-9};
    static const adcs_real v[ADCS_NX] = {1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2,
                                         1,    1,    1,    5e-6, 5e-6, 5e-6};
    adcs_config c;
    memset(&c, 0, sizeof c);
    c.variant = variant;
    c.dt = 0.1;
    c.process_dt = 0.1;
    c.inertia[0] = c.inertia[4] = 0.0418666;
    c.inertia[8] = 0.0066666;
    c.max_dipole[0] = c.max_dipole[1] = 0.13;
    c.max_dipole[2] = 0.082;
    c.flae_weight_sun = c.flae_weight_mag = 1;
    c.flae_tolerance = 1e-3;
    c.flae_max_iterations = 20;
    c.ukf_alpha = 1;
    memcpy(c.p_x0, p_x0, sizeof p_x0);
    memcpy(c.w_x, w_x, sizeof w_x);
    memcpy(c.v, v, sizeof v);
    for (int i = 0; i < 3; i++) {
        c.p_b0[i] = 1e-12;
        c.p_b0[3 + i] = 1e-7;
        c.w_b[i] = 1e-11;
        c.w_b[3 + i] = 1e-6;
    }
    c.sigma_gyro_noise = 1e-5;
    c.sigma_gyro_bias_walk = 3e-6;
    c.sigma_mag_noise = 0.1;
    c.sigma_mag_bias_walk = 1e-3;
    c.sigma_sun_noise = 2e-4;
    c.p0_attitude = 0.05;
    c.p0_gyro_bias = 5e-3;
    c.p0_mag_bias = 2;
    c.fault_nis_threshold = 25;
    c.fault_steps = 3;
    c.fault_gyro_bias_sigma = 0.1;
    c.fault_mag_bias_sigma = 50;
    c.fault_attitude_sigma = 0.05;
    c.control_enabled = control;
    c.bdot_gain = 4e4;
    c.kp = 2e-5;
    c.kd = 2e-3;
    c.q_target[3] = 1;
    c.detumble_enter_rate = 0.0698;
    c.detumble_exit_rate = 0.0175;
    c.settle_time = 1;
    return c;
}

static void to_input(const double r[19], adcs_input *in)
{
    memset(in, 0, sizeof *in);
    in->t = (adcs_real)r[0];
    for (size_t i = 0; i < 3; i++) {
        in->mag_b[i] = (adcs_real)r[1 + i];
        in->gyro_b[i] = (adcs_real)r[5 + i];
        in->sun_b[i] = (adcs_real)r[9 + i];
        in->sun_ref_i[i] = (adcs_real)r[13 + i];
        in->mag_ref_i[i] = (adcs_real)r[16 + i];
    }
    in->mag_valid = r[4] != 0.0;
    in->gyro_valid = r[8] != 0.0;
    in->sun_valid = r[12] != 0.0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    static void *mem;
    if (mem == NULL) {
        mem = aligned_alloc(alignof(max_align_t), (adcs_state_size() + 63) / 64 * 64);
    }
    if (size < 1 || mem == NULL) {
        return 0;
    }
    adcs_state *state = (adcs_state *)mem;
    const adcs_config cfg = base_config(data[0] & 1u, (data[0] & 2u) != 0);
    if (adcs_init(state, &cfg) != 0) {
        abort();
    }
    double fields[19];
    memcpy(fields, BASE_INPUT, sizeof fields);
    for (size_t pos = 1; pos + 9 <= size; pos += 9) {
        const unsigned op = data[pos] % 20u;
        if (op < 19) {
            memcpy(&fields[op], &data[pos + 1], sizeof(double));
        }
        adcs_input in;
        adcs_output out;
        to_input(fields, &in);
        if (adcs_step(state, &in, &out) != 0 || out.mode > ADCS_MODE_SAFE ||
            (out.status & ~STATUS_MASK) != 0) {
            abort();
        }
        const double n2 = (double)(out.q_est[0] * out.q_est[0] + out.q_est[1] * out.q_est[1] +
                                   out.q_est[2] * out.q_est[2] + out.q_est[3] * out.q_est[3]);
        const bool all_nan = isnan(out.q_est[0]) && isnan(out.q_est[1]) && isnan(out.q_est[2]) &&
                             isnan(out.q_est[3]);
        if (!all_nan && !(fabs(n2 - 1.0) < 1e-6)) {
            abort();
        }
        /* never command a NaN or out-of-range dipole to the coils */
        for (size_t i = 0; i < 3; i++) {
            if (!(fabs((double)out.m_cmd[i]) <= (double)cfg.max_dipole[i] * (1.0 + 1e-12))) {
                abort();
            }
        }
        fields[0] += 0.1;
    }
    return 0;
}
