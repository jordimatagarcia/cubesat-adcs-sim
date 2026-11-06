#include "flae.h"

#include <math.h>
#include <stddef.h>

#include "linalg.h"

#define MIN_SIN_ANGLE ((adcs_real)1e-3) /* closer to parallel carries no attitude information */

int flae_null_vector(adcs_real n[16], adcs_real out[4])
{
    size_t cols[4] = {0, 1, 2, 3};
    adcs_real scale = (adcs_real)0;
    for (size_t i = 0; i < 16; i++) {
        const adcs_real v = LA_FABS(n[i]);
        if (v > scale) {
            scale = v;
        }
    }
    if (scale == (adcs_real)0) {
        scale = (adcs_real)1;
    }
    for (size_t k = 0; k < 3; k++) {
        /* largest remaining element, first in row-major order */
        size_t pi = k, pj = k;
        adcs_real best = -(adcs_real)1;
        for (size_t i = k; i < 4; i++) {
            for (size_t j = k; j < 4; j++) {
                const adcs_real v = LA_FABS(LA_AT(n, 4, i, j));
                if (v > best) {
                    best = v;
                    pi = i;
                    pj = j;
                }
            }
        }
        if (!(best >= (adcs_real)1e-9 * scale)) {
            return -1;
        }
        for (size_t j = 0; j < 4; j++) {
            const adcs_real t = LA_AT(n, 4, k, j);
            LA_AT(n, 4, k, j) = LA_AT(n, 4, pi, j);
            LA_AT(n, 4, pi, j) = t;
        }
        for (size_t i = 0; i < 4; i++) {
            const adcs_real t = LA_AT(n, 4, i, k);
            LA_AT(n, 4, i, k) = LA_AT(n, 4, i, pj);
            LA_AT(n, 4, i, pj) = t;
        }
        const size_t tc = cols[k];
        cols[k] = cols[pj];
        cols[pj] = tc;
        for (size_t i = k + 1; i < 4; i++) {
            const adcs_real f = LA_AT(n, 4, i, k) / LA_AT(n, 4, k, k);
            for (size_t j = 0; j < 4; j++) {
                LA_AT(n, 4, i, j) -= f * LA_AT(n, 4, k, j);
            }
        }
    }
    /* Upper-triangular in the first three rows; the last (pivoted) unknown is free. */
    adcs_real x[4] = {0, 0, 0, 1};
    for (size_t kk = 3; kk-- > 0;) {
        adcs_real s = (adcs_real)0;
        for (size_t p = kk + 1; p < 4; p++) {
            s += LA_AT(n, 4, kk, p) * x[p];
        }
        x[kk] = -s / LA_AT(n, 4, kk, kk);
    }
    for (size_t i = 0; i < 4; i++) {
        out[cols[i]] = x[i];
    }
    return 0;
}

int flae2(const adcs_real v_body[6], const adcs_real v_ref[6], const adcs_real weights[2],
          const adcs_real *q_prev, adcs_real tolerance, uint32_t max_iterations, flae_result *out)
{
    const adcs_real wsum = weights[0] + weights[1];
    if (!(wsum > (adcs_real)0)) {
        return -1;
    }
    const adcs_real a[2] = {weights[0] / wsum, weights[1] / wsum};
    adcs_real c[3];
    v3_cross(&v_body[0], &v_body[3], c);
    if (!(v3_norm(c) >= MIN_SIN_ANGLE)) {
        return -1;
    }

    /* Attitude profile matrix B = Σ a b rᵀ; H_x, H_y, H_z are its columns. */
    adcs_real b[9];
    for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 3; k++) {
            b[j * 3 + k] = a[0] * v_body[j] * v_ref[k] + a[1] * v_body[3 + j] * v_ref[3 + k];
        }
    }
    const adcs_real hx[3] = {b[0], b[3], b[6]};
    const adcs_real hy[3] = {b[1], b[4], b[7]};
    const adcs_real hz[3] = {b[2], b[5], b[8]};
    /* eq. 105 */
    const adcs_real w[16] = {
        hx[0] + hy[1] + hz[2], -hy[2] + hz[1],        -hz[0] + hx[2],        -hx[1] + hy[0],
        -hy[2] + hz[1],        hx[0] - hy[1] - hz[2], hx[1] + hy[0],         hx[2] + hz[0],
        -hz[0] + hx[2],        hx[1] + hy[0],         hy[1] - hx[0] - hz[2], hy[2] + hz[1],
        -hx[1] + hy[0],        hx[2] + hz[0],         hy[2] + hz[1],         hz[2] - hy[1] - hx[0],
    };
    /* eq. 107: det(W − λI) = λ⁴ + τ1 λ² + τ2 λ + τ3 */
    adcs_real bb = (adcs_real)0;
    for (size_t i = 0; i < 9; i++) {
        bb += b[i] * b[i];
    }
    const adcs_real tau1 = (adcs_real)-2 * bb;
    const adcs_real tau2 =
        (adcs_real)8 * (hx[2] * hy[1] * hz[0] - hx[1] * hy[2] * hz[0] - hx[2] * hy[0] * hz[1] +
                        hx[0] * hy[2] * hz[1] + hx[1] * hy[0] * hz[2] - hx[0] * hy[1] * hz[2]);
    adcs_real work[16];
    const adcs_real tau3 = la_det(4, w, work);

    adcs_real lam = (adcs_real)1;
    uint32_t it = 0;
    int converged = 0;
    while (it < max_iterations) {
        it++;
        const adcs_real l2 = lam * lam;
        const adcs_real f = l2 * l2 + tau1 * l2 + tau2 * lam + tau3;
        const adcs_real df = (adcs_real)4 * l2 * lam + (adcs_real)2 * tau1 * lam + tau2;
        if (df == (adcs_real)0) {
            return -1;
        }
        const adcs_real step = f / df;
        lam -= step;
        if (LA_FABS(step) < tolerance) {
            converged = 1;
            break;
        }
    }
    if (!converged) {
        return -1;
    }

    adcs_real n[16];
    la_copy(16, n, w);
    for (size_t i = 0; i < 4; i++) {
        LA_AT(n, 4, i, i) -= lam;
    }
    adcs_real v[4];
    if (flae_null_vector(n, v) != 0) {
        return -1;
    }
    adcs_real q[4] = {v[1], v[2], v[3], v[0]}; /* scalar-first eigenvector -> scalar-last */
    const adcs_real qn = LA_SQRT(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (size_t i = 0; i < 4; i++) {
        q[i] /= qn;
    }
    int flip;
    if (q_prev != NULL) {
        flip = la_all_finite(4, q_prev) &&
               (q[0] * q_prev[0] + q[1] * q_prev[1] + q[2] * q_prev[2] + q[3] * q_prev[3]) <
                   (adcs_real)0;
    } else {
        flip = q[3] < (adcs_real)0;
    }
    for (size_t i = 0; i < 4; i++) {
        out->q[i] = flip ? -q[i] : q[i];
    }
    out->lambda_max = lam;
    out->iterations = it;
    return 0;
}
