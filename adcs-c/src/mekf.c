#include "mekf.h"

#include <math.h>
#include <stddef.h>

#include "linalg.h"
#include "quat.h"

#define NX MEKF_NX
#define NY MEKF_NY
#define UNAVAILABLE_VARIANCE ((adcs_real)1e12)

void mekf_configure(mekf *f, adcs_real sigma_gyro_noise, adcs_real sigma_gyro_bias_walk,
                    adcs_real sigma_mag_noise, adcs_real sigma_mag_bias_walk,
                    adcs_real sigma_sun_noise)
{
    f->sigma_gyro_noise = sigma_gyro_noise;
    f->sigma_gyro_bias_walk = sigma_gyro_bias_walk;
    f->sigma_mag_bias_walk = sigma_mag_bias_walk;
    for (size_t i = 0; i < 3; i++) {
        f->r_diag[i] = sigma_sun_noise * sigma_sun_noise;
        f->r_diag[3 + i] = sigma_mag_noise * sigma_mag_noise;
    }
    const adcs_real q0[4] = {0, 0, 0, 1};
    adcs_real p0[NX];
    for (size_t i = 0; i < NX; i++) {
        p0[i] = (adcs_real)1;
    }
    mekf_reset(f, q0, p0);
}

void mekf_reset(mekf *f, const adcs_real q0[4], const adcs_real p0_diag[NX])
{
    la_copy(4, f->q_ref, q0);
    quat_normalize(f->q_ref);
    la_zero(NX, f->x);
    la_copy(NX, f->p0_diag, p0_diag);
    la_zero(NX * NX, f->p);
    for (size_t i = 0; i < NX; i++) {
        LA_AT(f->p, NX, i, i) = p0_diag[i];
    }
    f->has_update = false;
}

/* Exact Φ11 = exp(−[ω×]dt) and Φ12 = −∫₀^dt exp(−[ω×]s) ds for a constant rate. */
static void rotation_blocks(const adcs_real w[3], adcs_real dt, adcs_real phi11[9],
                            adcs_real phi12[9])
{
    adcs_real wx[9], wx2[9];
    v3_skew(w, wx);
    const adcs_real rate = v3_norm(w);
    if (rate < (adcs_real)1e-9) {
        for (size_t i = 0; i < 9; i++) {
            phi11[i] = -wx[i] * dt;
            phi12[i] = (adcs_real)0.5 * wx[i] * dt * dt;
        }
        for (size_t i = 0; i < 3; i++) {
            LA_AT(phi11, 3, i, i) += (adcs_real)1;
            LA_AT(phi12, 3, i, i) -= dt;
        }
        return;
    }
    const adcs_real th = rate * dt;
    const adcs_real s = LA_SIN(th);
    const adcs_real c = LA_COS(th);
    const adcs_real rate2 = rate * rate;
    adcs_real scaled[9];
    for (size_t i = 0; i < 9; i++) {
        scaled[i] = (((adcs_real)1 - c) / rate2) * wx[i];
    }
    la_mul(3, 3, 3, scaled, wx, phi11); /* ((1 − cos θ)/ω²) [ω×] [ω×] */
    la_mul(3, 3, 3, wx, wx, wx2);
    for (size_t i = 0; i < 9; i++) {
        phi11[i] -= s / rate * wx[i];
        phi12[i] = wx[i] * ((adcs_real)1 - c) / rate2 - wx2[i] * (th - s) / (rate2 * rate);
    }
    for (size_t i = 0; i < 3; i++) {
        LA_AT(phi11, 3, i, i) += (adcs_real)1;
        LA_AT(phi12, 3, i, i) -= dt;
    }
}

bool mekf_step(mekf *f, const adcs_real gyro[3], adcs_real dt, const adcs_real sun_b[3],
               const adcs_real sun_i[3], bool sun_available, const adcs_real mag_ut[3],
               const adcs_real mag_i_ut[3], bool mag_available)
{
    mekf_work *k = &f->work;
    adcs_real *phi = k->phi, *tmp = k->tmp, *p_pred = k->p_pred, *ikh = k->ikh;
    adcs_real *p_upd = k->p_upd, *krk = k->krk, *h = k->h, *hp = k->hp, *pht = k->pht;
    adcs_real *gain = k->gain, *kr = k->kr;

    /* ---- propagation ------------------------------------------------------------------ */
    adcs_real w[3], dq[4], qn[4];
    for (size_t i = 0; i < 3; i++) {
        w[i] = gyro[i] - f->x[MEKF_GYRO + i];
    }
    quat_from_rate(w, dt, dq);
    quat_mul(dq, f->q_ref, qn);
    la_copy(4, f->q_ref, qn);

    adcs_real phi11[9], phi12[9];
    rotation_blocks(w, dt, phi11, phi12);
    la_eye(NX, phi);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            LA_AT(phi, NX, MEKF_ATT + i, MEKF_ATT + j) = LA_AT(phi11, 3, i, j);
            LA_AT(phi, NX, MEKF_ATT + i, MEKF_GYRO + j) = LA_AT(phi12, 3, i, j);
        }
    }
    const adcs_real sv2 = f->sigma_gyro_noise * f->sigma_gyro_noise * dt;
    const adcs_real su2 = f->sigma_gyro_bias_walk * f->sigma_gyro_bias_walk;
    const adcs_real dt2 = dt * dt;
    la_mul(NX, NX, NX, phi, f->p, tmp);
    la_mul_bt(NX, NX, NX, tmp, phi, p_pred);
    for (size_t i = 0; i < 3; i++) {
        LA_AT(p_pred, NX, MEKF_ATT + i, MEKF_ATT + i) += sv2 * dt + su2 * (dt2 * dt) / (adcs_real)3;
        LA_AT(p_pred, NX, MEKF_ATT + i, MEKF_GYRO + i) += -(su2 * dt2 / (adcs_real)2);
        LA_AT(p_pred, NX, MEKF_GYRO + i, MEKF_ATT + i) += -(su2 * dt2 / (adcs_real)2);
        LA_AT(p_pred, NX, MEKF_GYRO + i, MEKF_GYRO + i) += su2 * dt;
        LA_AT(p_pred, NX, MEKF_MAG + i, MEKF_MAG + i) +=
            f->sigma_mag_bias_walk * f->sigma_mag_bias_walk * dt;
    }

    /* ---- update ----------------------------------------------------------------------- */
    adcs_real a[9], sun_hat[3], mag_body[3];
    quat_to_dcm(f->q_ref, a);
    la_mul_vec(3, 3, a, sun_i, sun_hat);
    la_mul_vec(3, 3, a, mag_i_ut, mag_body);
    adcs_real y_hat[NY], y[NY];
    bool avail[NY];
    for (size_t i = 0; i < 3; i++) {
        y_hat[i] = sun_hat[i];
        y_hat[3 + i] = mag_body[i] + f->x[MEKF_MAG + i];
        avail[i] = sun_available;
        avail[3 + i] = mag_available;
        y[i] = sun_available ? sun_b[i] : y_hat[i];
        y[3 + i] = mag_available ? mag_ut[i] : y_hat[3 + i];
    }
    la_zero(NY * NX, h);
    adcs_real sk[9];
    v3_skew(sun_hat, sk);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            LA_AT(h, NX, i, MEKF_ATT + j) = LA_AT(sk, 3, i, j);
        }
    }
    v3_skew(mag_body, sk);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            LA_AT(h, NX, 3 + i, MEKF_ATT + j) = LA_AT(sk, 3, i, j);
        }
        LA_AT(h, NX, 3 + i, MEKF_MAG + i) = (adcs_real)1;
    }
    adcs_real r[NY];
    for (size_t i = 0; i < NY; i++) {
        r[i] = avail[i] ? f->r_diag[i] : UNAVAILABLE_VARIANCE;
    }
    adcs_real s[NY * NY];
    la_mul(NY, NX, NX, h, p_pred, hp);
    la_mul_bt(NY, NX, NY, hp, h, s);
    for (size_t i = 0; i < NY; i++) {
        LA_AT(s, NY, i, i) += r[i];
    }
    la_mul_bt(NX, NX, NY, p_pred, h, pht);
    const int solved = la_right_divide(NX, NY, pht, s, gain, k->solve);
    adcs_real innov[NY], dx[NX];
    for (size_t i = 0; i < NY; i++) {
        innov[i] = y[i] - y_hat[i];
    }
    la_mul_vec(NX, NY, gain, innov, dx);
    /* Joseph form: (I − KH) P (I − KH)ᵀ + K R Kᵀ */
    la_mul(NX, NY, NX, gain, h, ikh);
    for (size_t i = 0; i < NX * NX; i++) {
        ikh[i] = -ikh[i];
    }
    for (size_t i = 0; i < NX; i++) {
        LA_AT(ikh, NX, i, i) += (adcs_real)1;
    }
    la_mul(NX, NX, NX, ikh, p_pred, tmp);
    la_mul_bt(NX, NX, NX, tmp, ikh, p_upd);
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            LA_AT(kr, NY, i, j) = LA_AT(gain, NY, i, j) * r[j];
        }
    }
    la_mul_bt(NX, NY, NX, kr, gain, krk);
    for (size_t i = 0; i < NX * NX; i++) {
        p_upd[i] += krk[i];
    }
    la_symmetrize(NX, p_upd);

    f->has_update = true;
    for (size_t i = 0; i < NY; i++) {
        f->available[i] = avail[i];
        f->innovation[i] = avail[i] ? innov[i] : (adcs_real)NAN;
    }
    la_copy(NY * NY, f->innovation_cov, s);

    if (solved != 0 || !la_all_finite(NX, dx) || !la_all_finite(NX * NX, p_upd)) {
        la_zero(NX * NX, f->p);
        for (size_t i = 0; i < NX; i++) {
            LA_AT(f->p, NX, i, i) = f->p0_diag[i];
        }
        return false;
    }
    adcs_real e[4] = {(adcs_real)0.5 * dx[MEKF_ATT], (adcs_real)0.5 * dx[MEKF_ATT + 1],
                      (adcs_real)0.5 * dx[MEKF_ATT + 2], (adcs_real)1};
    quat_normalize(e);
    quat_mul(e, f->q_ref, qn);
    quat_normalize(qn);
    la_copy(4, f->q_ref, qn);
    for (size_t i = 0; i < 3; i++) {
        f->x[MEKF_GYRO + i] += dx[MEKF_GYRO + i];
        f->x[MEKF_MAG + i] += dx[MEKF_MAG + i];
    }
    la_copy(NX * NX, f->p, p_upd);
    return true;
}

void mekf_inflate(mekf *f, adcs_real attitude, adcs_real gyro_bias, adcs_real mag_bias)
{
    for (size_t i = 0; i < 3; i++) {
        LA_AT(f->p, NX, MEKF_ATT + i, MEKF_ATT + i) += attitude * attitude;
        LA_AT(f->p, NX, MEKF_GYRO + i, MEKF_GYRO + i) += gyro_bias * gyro_bias;
        LA_AT(f->p, NX, MEKF_MAG + i, MEKF_MAG + i) += mag_bias * mag_bias;
    }
}

adcs_real mekf_nis(const mekf *f, int block)
{
    const size_t r0 = block == 0 ? 0 : 3;
    if (!f->has_update || !f->available[r0] || !f->available[r0 + 1] || !f->available[r0 + 2]) {
        return (adcs_real)NAN;
    }
    adcs_real s[9], e[3];
    for (size_t i = 0; i < 3; i++) {
        e[i] = f->innovation[r0 + i];
        for (size_t j = 0; j < 3; j++) {
            LA_AT(s, 3, i, j) = LA_AT(f->innovation_cov, NY, r0 + i, r0 + j);
        }
    }
    adcs_real x[3] = {e[0], e[1], e[2]};
    if (la_solve(3, 1, s, x) != 0) {
        return (adcs_real)NAN;
    }
    return v3_dot(e, x);
}
