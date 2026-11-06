/*
 * mekf.h: multiplicative extended Kalman filter (internal), Markley & Crassidis (2014) §6.2.
 * Port of src/adcssim/adcs_ref/mekf.py.
 *
 * State (9): attitude error δθ (always folded into q_ref after an update), gyro bias (rad/s),
 * magnetometer bias (µT). The gyro is the rate input; the sun and magnetometer vectors are
 * measured against the attitude: y_sun = A(q) s_i, y_B = A(q) B_i + β_m.
 */
#ifndef ADCS_MEKF_H
#define ADCS_MEKF_H

#include <stdbool.h>

#include "adcs.h"

#define MEKF_NX ADCS_MEKF_NX
#define MEKF_NY 6
#define MEKF_ATT 0
#define MEKF_GYRO 3
#define MEKF_MAG 6

typedef struct {
    adcs_real phi[MEKF_NX * MEKF_NX];
    adcs_real tmp[MEKF_NX * MEKF_NX];
    adcs_real p_pred[MEKF_NX * MEKF_NX];
    adcs_real ikh[MEKF_NX * MEKF_NX];
    adcs_real p_upd[MEKF_NX * MEKF_NX];
    adcs_real krk[MEKF_NX * MEKF_NX];
    adcs_real h[MEKF_NY * MEKF_NX];
    adcs_real hp[MEKF_NY * MEKF_NX];
    adcs_real pht[MEKF_NX * MEKF_NY];
    adcs_real gain[MEKF_NX * MEKF_NY];
    adcs_real kr[MEKF_NX * MEKF_NY];
    adcs_real solve[MEKF_NY * MEKF_NY + MEKF_NY * MEKF_NX];
} mekf_work;

typedef struct {
    adcs_real sigma_gyro_noise;
    adcs_real sigma_gyro_bias_walk;
    adcs_real sigma_mag_bias_walk;
    adcs_real r_diag[MEKF_NY];
    adcs_real q_ref[4];
    adcs_real x[MEKF_NX];
    adcs_real p[MEKF_NX * MEKF_NX];
    adcs_real p0_diag[MEKF_NX];
    /* last update, for the fault detector */
    bool has_update;
    bool available[MEKF_NY];
    adcs_real innovation[MEKF_NY];
    adcs_real innovation_cov[MEKF_NY * MEKF_NY];
    mekf_work work;
} mekf;

void mekf_configure(mekf *f, adcs_real sigma_gyro_noise, adcs_real sigma_gyro_bias_walk,
                    adcs_real sigma_mag_noise, adcs_real sigma_mag_bias_walk,
                    adcs_real sigma_sun_noise);
void mekf_reset(mekf *f, const adcs_real q0[4], const adcs_real p0_diag[MEKF_NX]);

/* Propagate over dt with the rate gyro (rad/s), then update with the vectors (unit sun,
 * field in µT). Returns false if the result was not finite and the covariance was reset. */
bool mekf_step(mekf *f, const adcs_real gyro[3], adcs_real dt, const adcs_real sun_b[3],
               const adcs_real sun_i[3], bool sun_available, const adcs_real mag_ut[3],
               const adcs_real mag_i_ut[3], bool mag_available);

/* Add variance to the diagonal (standard deviations given). */
void mekf_inflate(mekf *f, adcs_real attitude, adcs_real gyro_bias, adcs_real mag_bias);

/* Normalised innovation squared of the sun (block 0) or magnetometer (block 1) in the last
 * update; NaN if there was none or the measurement was unavailable. */
adcs_real mekf_nis(const mekf *f, int block);

#endif /* ADCS_MEKF_H */
