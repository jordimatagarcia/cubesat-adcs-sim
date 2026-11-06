/*
 * tsukf.h: two-stage unscented Kalman filter (internal), Chen et al. as derived in the
 * thesis (§2.4.3, eqs. 49–68). Port of src/adcssim/adcs_ref/tsukf.py, specialised to the
 * thesis ADCS: state x (13), bias b (6), measurement y = h(x) + F_s b with h(x) = x, F_a = 0,
 * and F_s mapping the magnetometer bias onto B and the gyro bias onto ω.
 *
 * Every temporary lives in the struct, so the filter uses a fixed amount of memory and very
 * little stack.
 */
#ifndef ADCS_TSUKF_H
#define ADCS_TSUKF_H

#include <stdbool.h>

#include "adcs.h"

#define TS_NX ADCS_NX
#define TS_NB ADCS_NB
#define TS_NY ADCS_NX
#define TS_NS (2 * TS_NX + 1) /* sigma points */

/* State layout */
#define TS_Q 0
#define TS_SUN 4
#define TS_MAG 7
#define TS_W 10

/* Process model applied to each sigma point (one row of TS_NX values, in place). */
typedef void (*tsukf_process_fn)(adcs_real x[TS_NX], const void *ctx);

typedef struct {
    adcs_real px[TS_NX * TS_NX];
    adcs_real pb[TS_NB * TS_NB];
    adcs_real beta[TS_NX * TS_NB];
    adcs_real lower[TS_NX * TS_NX];
    adcs_real chi[TS_NS * TS_NX];
    adcs_real diff[TS_NX * TS_NX];
    adcs_real m[TS_NX * TS_NX];
    adcs_real n[TS_NX * TS_NX];
    adcs_real r[TS_NX * TS_NB];
    adcs_real h[TS_NY * TS_NB];
    adcs_real beta_pred[TS_NX * TS_NB];
    adcs_real beta_upd[TS_NX * TS_NB];
    adcs_real t1[TS_NX * TS_NB];
    adcs_real t2[TS_NX * TS_NB];
    adcs_real px_pred[TS_NX * TS_NX];
    adcs_real px_t[TS_NX * TS_NX];
    adcs_real px_upd[TS_NX * TS_NX];
    adcs_real py_t[TS_NY * TS_NY];
    adcs_real p_y[TS_NY * TS_NY];
    adcs_real k_t[TS_NX * TS_NY];
    adcs_real k_b[TS_NB * TS_NY];
    adcs_real p_by[TS_NB * TS_NY];
    adcs_real a[TS_NX * TS_NX];
    adcs_real b[TS_NX * TS_NX];
    adcs_real c[TS_NX * TS_NX];
    adcs_real pb_pred[TS_NB * TS_NB];
    adcs_real pb_inv[TS_NB * TS_NB];
    adcs_real pb_upd[TS_NB * TS_NB];
    adcs_real six_a[TS_NB * TS_NB];
    adcs_real nb_ny[TS_NB * TS_NY];
    adcs_real solve[2 * TS_NX * TS_NX];
} tsukf_work;

typedef struct {
    /* weights of the unscented transform */
    adcs_real xi, wm0, wc0, wi;
    adcs_real w_x[TS_NX], w_b[TS_NB], v[TS_NY]; /* diagonals */
    adcs_real px0[TS_NX], pb0[TS_NB];           /* initial covariance diagonals */
    /* estimate */
    adcs_real x[TS_NX];
    adcs_real px[TS_NX * TS_NX];
    adcs_real b[TS_NB];
    adcs_real pb[TS_NB * TS_NB];
    adcs_real beta[TS_NX * TS_NB];
    tsukf_work work;
} tsukf;

void tsukf_configure(tsukf *f, adcs_real alpha, const adcs_real w_x[TS_NX],
                     const adcs_real w_b[TS_NB], const adcs_real v[TS_NY]);
void tsukf_reset(tsukf *f, const adcs_real x0[TS_NX], const adcs_real px0[TS_NX],
                 const adcs_real b0[TS_NB], const adcs_real pb0[TS_NB]);

/* One predict + update. Unavailable measurements get an effectively infinite variance and
 * the predicted value. q is renormalised after the prediction, q and the sun vector at the
 * output. Returns false if a covariance had to be reset. */
bool tsukf_step(tsukf *f, tsukf_process_fn process, const void *ctx, const adcs_real y[TS_NY],
                const bool available[TS_NY]);

#endif /* ADCS_TSUKF_H */
